"""Module/algorithm regression checks for Project 3 independent modular SRC."""
import gzip
import json
import math
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e4_03_term_weighting import WEIGHTINGS, tf_weight, WEIGHTING_FUNCTIONS
from e7_01_retrieval_evaluation import evaluate_ranking, evaluate_all, make_qrels, load_rules
from e7_02_weighting_experiments import compare_rankings, compare_normalization
from m02_text_processor import tokens, idf
from m08_controller import SearchIndex
from m01_corpus_reader import read_corpus, sha256
from project3_config import CORPUS_PATH, ROOT


def test_weighting_formulas_and_registry():
    assert tuple(WEIGHTING_FUNCTIONS) == WEIGHTINGS
    assert tf_weight(0, "Raw TF", .5) == 0
    assert tf_weight(10, "Raw TF", .5) == 10
    assert tf_weight(10, "Log TF", .5) == pytest.approx(2.0)
    assert tf_weight(10, "IDF Only", .5) == pytest.approx(.5)
    assert tf_weight(10, "Raw TF-IDF", .5) == pytest.approx(5.0)
    assert tf_weight(10, "Log TF-IDF", .5) == pytest.approx(1.0)
    assert idf(1000, 100) == pytest.approx(1.0)
    assert idf(1000, 1000) == pytest.approx(0.0)
    assert idf(1000, 1) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        tf_weight(1, "Not a model", .5)


def test_project2_preprocessing_compatibility():
    assert tokens("GLP-1 diabetes.", "B") == ["glp-1", "diabetes"]
    assert tokens("The GLP-1 and insulin", "C") == ["glp-1", "insulin"]
    assert tokens("The GLP-1 and insulin", "A")[0] == "the"
    assert "the" not in tokens("The GLP-1", "D")
    assert tokens("treatments", "D") == ["treatment"]
    with pytest.raises(ValueError):
        tokens("hello", "invalid")


def mini_corpus(tmp_path):
    docs = [("1", "alpha alpha alpha beta"), ("2", "alpha beta gamma"),
            ("3", "beta gamma gamma"), ("4", "gamma delta")]
    file = tmp_path / "mini.jsonl.gz"
    with gzip.open(file, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"record_type": "project2_corpus_header", "corpus_sha256": "test"}) + "\n")
        for pmid, abstract in docs:
            output.write(json.dumps({"pmid": pmid, "title": "Doc " + pmid,
                                     "abstract_clean": abstract}) + "\n")
    return file


def test_corpus_input_validation(tmp_path):
    docs, header = read_corpus(mini_corpus(tmp_path))
    assert len(docs) == 4 and header["corpus_sha256"] == "test"
    malformed = tmp_path / "bad.jsonl.gz"
    with gzip.open(malformed, "wt", encoding="utf-8") as out:
        out.write(json.dumps({"pmid": "1", "abstract": "foo"}) + "\n")
    with pytest.raises(ValueError, match="header"):
        read_corpus(malformed)


def test_vectors_cosine_l2_ranking_and_zero_weights(tmp_path):
    idx = SearchIndex.build(mini_corpus(tmp_path), condition="B")
    assert len(idx.docs) == 4
    assert idx.df["beta"] == 3
    assert idx.cf["alpha"] == 4
    assert idx.idfs[idx.vocabulary["gamma"]] == pytest.approx(math.log10(4/3))
    for method in WEIGHTINGS:
        assert idx.matrices[method].shape == (4, 4)
        cosine = idx.search("alpha gamma", method, "Cosine", 10)
        l2 = idx.search("alpha gamma", method, "L2 Dot Product", 10)
        assert [h["pmid"] for h in cosine["hits"]] == [h["pmid"] for h in l2["hits"]]
        assert [h["score"] for h in cosine["hits"]] == pytest.approx(
            [h["score"] for h in l2["hits"]], abs=1e-12)
    assert idx.search("zzzzzz")["hits"] == []
    assert idx.search("zzzzzz")["oov"] == ["zzzzzz"]
    assert len(idx.search("alpha")["hits"]) == 2
    with pytest.raises(ValueError):
        idx.search("alpha", metric="Unknown")
    with pytest.raises(ValueError):
        idx.search("alpha", limit=0)
    assert idx.term_stats("alpha")[0]["DF"] == 2
    assert idx.explain_terms("alpha", 0)[0]["Query TF"] > 0


def test_retrieval_metrics_and_qrels_assumptions():
    hits = [{"doc_index": 2}, {"doc_index": 1}, {"doc_index": 3}]
    r = evaluate_ranking(hits, {1, 3}, k=10)
    assert r["Precision@10"] == pytest.approx(.2)
    assert r["Recall@10"] == pytest.approx(1.0)
    assert r["MRR@10"] == pytest.approx(.5)
    assert 0 < r["nDCG@10"] < 1
    assert evaluate_ranking(hits, set())["MRR@10"] is None


@pytest.fixture(scope="module")
def actual_index():
    return SearchIndex.build()


def test_frozen_project2_corpus(actual_index):
    s = actual_index.summary()
    assert s["documents"] == s["unique_pmids"] == 1000
    assert s["project2_header_sha256"] == "7c3739fab90f225e628f47fe737596bd6f2059e35300005801267c1152ae4756"
    assert s["corpus_sha256"] == sha256(CORPUS_PATH)
    assert s["vocabulary"] == 14588 and s["tokens"] == 136195


def test_full_evaluation_and_reproducibility(actual_index):
    rules = load_rules()
    assert len(rules) == 10
    qrels = make_qrels(actual_index, rules)
    assert all(qrels[r["id"]] for r in rules)
    rows, means, _ = evaluate_all(actual_index, rules)
    assert len(rows) == 50 and len(means) == 5
    assert [r["Weighting"] for r in means] == list(WEIGHTINGS)
    assert means[-1]["nDCG@10"] == pytest.approx(0.9394434793486648, abs=1e-12)
    assert all(0 <= row["nDCG@10"] <= 1 for row in rows)
    assert all(0 <= row["MRR@10"] <= 1 for row in rows)


def test_cosine_dot_and_experiment_extension(actual_index):
    for method in WEIGHTINGS:
        experiment = compare_normalization(actual_index, "GLP-1 kidney diabetes", method)
        assert experiment["cosine_l2_same_order"]
        assert experiment["max_absolute_error"] < 1e-10
    ranks = compare_rankings(actual_index, "semaglutide weight loss")
    assert ranks and "Raw TF" in ranks[0] and "Raw TF-IDF" in ranks[0]


def test_modules_do_not_import_ui():
    """Early warning for accidental coupling when future E-modules are added."""
    for p in (ROOT / "src").glob("*.py"):
        if p.name == "m09_app.py":
            continue
        text = p.read_text(encoding="utf-8")
        assert "import streamlit" not in text, p.name


def test_optional_e8_vector_projection(tmp_path):
    """SVD is query-transformable, deterministic and does not alter IR ranking."""
    from e8_01_vector_projection import fit_document_projection, project_query
    from m04_query_processor import query_vector
    mini = SearchIndex.build(mini_corpus(tmp_path), condition="B")
    before = mini.search("alpha gamma", "Raw TF-IDF", "Cosine", 10)
    p = fit_document_projection(mini, "Raw TF-IDF")
    p2 = fit_document_projection(mini, "Raw TF-IDF")
    assert p.coordinates.shape == (len(mini.docs), 2)
    assert np.isfinite(p.coordinates).all()
    assert np.allclose(p.coordinates, p2.coordinates, atol=1e-12)
    xy, missing = project_query(mini, p, "alpha gamma")
    assert not missing and xy.shape == (2,) and np.isfinite(xy).all()
    qvec, _, _ = query_vector(mini, "alpha gamma", "Raw TF-IDF")
    qnorm = float(np.sqrt(qvec.multiply(qvec).sum()))
    assert xy == pytest.approx(p.model.transform(qvec.multiply(1 / qnorm))[0])
    assert project_query(mini, p, "nonexistenttoken") == (None, ["nonexistenttoken"])
    assert before == mini.search("alpha gamma", "Raw TF-IDF", "Cosine", 10)
    with pytest.raises(ValueError, match="Unknown weighting"):
        fit_document_projection(mini, "Bad weighting")
    with pytest.raises(ValueError, match="different corpus"):
        project_query(SearchIndex.build(mini_corpus(tmp_path), condition="C"), p, "alpha")


def test_full_corpus_two_dimensional_projection(actual_index):
    from e8_01_vector_projection import fit_document_projection, project_query
    p = fit_document_projection(actual_index, "Raw TF-IDF")
    assert p.coordinates.shape == (1000, 2)
    assert 0 < sum(p.explained_variance_ratio) < 1
    xy, missing = project_query(actual_index, p, "semaglutide weight loss")
    assert xy is not None and not missing


def test_top_navigation_and_projection_ui_structure():
    """Static UI contract check; browser interaction is validated separately on Windows."""
    import ast
    from project3_config import ROOT
    app_text = (ROOT / "src" / "m09_app.py").read_text(encoding="utf-8")
    ast.parse(app_text)
    assert "project3_active_page" in app_text
    assert 'getattr(st, "segmented_control"' in app_text
    assert "with st.sidebar:" not in app_text
    for label in ("01 Retrieval", "02 Term Weighting", "03 Controlled Experiments",
                  "04 Evaluation", "05 Vector Projection"):
        assert label in app_text
    assert "cached_projection" in app_text
    assert 'st.plotly_chart(fig' in app_text
