"""Tests use original Project 2 Extension artifacts placed into Project 3."""
from pathlib import Path
import ast
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from m01_corpus_reader import read_corpus
from e3_01_word2vec_viewer import (model_inventory, read_training_report,
                                  load_trained_word2vec, compare_embeddings,
                                  normalized_single_token)
from project3_config import EMBEDDING_MODELS


def test_model_inventory_and_provenance_same_corpus():
    assert len(EMBEDDING_MODELS) == 3
    assert all(model_inventory().values())
    docs, header = read_corpus()
    report = read_training_report(expected_corpus_sha256=header["corpus_sha256"])
    assert len(docs) == report["corpus_stats"]["documents"] == 1000
    assert len(report["experiments"]) == 3
    with pytest.raises(ValueError, match="differs"):
        read_training_report(expected_corpus_sha256="0"*64)


@pytest.mark.parametrize("label", ["CBOW W5", "Skip-gram W5", "Skip-gram W2"])
def test_all_saved_word2vec_models_load_and_compute_real_neighbors(label):
    model = load_trained_word2vec(label)
    assert model.wv.vector_size == 100
    assert len(model.wv) == 7959
    r = compare_embeddings(model, "semaglutide", "glp-1", top_k=5)
    assert r["term_exists"] and r["other_exists"]
    assert len(r["neighbors"]) == 5
    assert all(-1 <= v["Cosine Similarity"] <= 1 for v in r["neighbors"])
    assert -1 <= r["pair_cosine"] <= 1
    assert len({v["Term"] for v in r["neighbors"]}) == 5


def test_oov_phrase_input_validation_and_missing_artifact(tmp_path):
    model = load_trained_word2vec("CBOW W5")
    r = compare_embeddings(model, "wordwhichdoesnotexist987", "semaglutide")
    assert not r["term_exists"] and r["pair_cosine"] is None
    r2 = compare_embeddings(model, "semaglutide", "wordwhichdoesnotexist987")
    assert not r2["other_exists"] and r2["pair_cosine"] is None
    with pytest.raises(ValueError, match="one token"):
        compare_embeddings(model, "weight loss")
    with pytest.raises(ValueError):
        compare_embeddings(model, "semaglutide", top_k=100)
    with pytest.raises(FileNotFoundError):
        load_trained_word2vec("CBOW W5", folder=tmp_path)
    with pytest.raises(ValueError, match="Unknown model"):
        load_trained_word2vec("CBOW W2")


def test_lexical_vs_semantic_actual_evidence_and_ui_contract():
    model = load_trained_word2vec("CBOW W5")
    assert compare_embeddings(model, "semaglutide", "liraglutide")["pair_cosine"] > 0.7
    root = Path(__file__).resolve().parents[1]
    ui = (root / "src/m09_app.py").read_text(encoding="utf-8")
    ast.parse(ui)
    assert "with right_panel:" in ui
    assert '"01-1 Weighting Comparison"' in ui
    assert 'read_training_report(expected_corpus_sha256=' in ui
    assert "Word2Vec pair cosine" in ui
    assert "compare_query_document_terms(" in ui
    assert "Word2Vec is not used for Project 3 retrieval" in ui
