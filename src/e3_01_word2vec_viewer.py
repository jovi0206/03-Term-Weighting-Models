"""E3 — Read-only view of Project 3 Word2Vec models trained on 3,000 Main Abstracts.

Training runs offline in e3_02_train_word2vec_3000.py; model loading and
word cosine remain separate from lexical retrieval and its evaluation. Model files are trusted local user-provided artifacts; loading a
Gensim model can deserialize Python objects, so never load untrusted models.
"""
from __future__ import annotations

from difflib import get_close_matches
import gzip
import json
from pathlib import Path

from project3_config import EMBEDDING_DIR, EMBEDDING_MODELS, EMBEDDING_REPORT_PATH, FORMAL_DOCUMENT_COUNT, CORPUS_PATH
from m01_corpus_reader import sha256


def model_inventory(folder=None):
    """Report exact required files without training or downloading anything."""
    base = Path(folder) if folder is not None else EMBEDDING_DIR
    return {label: (base / filename).is_file()
            for label, filename in EMBEDDING_MODELS.items()}


def read_training_report(path=None, expected_corpus_sha256=None):
    """Reject stale/mismatched Word2Vec training artifacts."""
    p = Path(path) if path is not None else EMBEDDING_REPORT_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Missing Project 3 3,000-document embedding report: {p}")
    with gzip.open(p, "rt", encoding="utf-8") as f:
        report = json.load(f)
    required = {"corpus_id", "corpus_sha256", "experiments", "corpus_stats"}
    if not required.issubset(report):
        raise ValueError("Invalid embedding report: provenance fields missing")
    if report.get("corpus_id") != "glp1_3000" or report.get("corpus_stats", {}).get("documents") != FORMAL_DOCUMENT_COUNT:
        raise ValueError("Word2Vec report is not from Project 3 formal 3,000-document corpus")
    if report.get("corpus_file_sha256") != sha256(CORPUS_PATH):
        raise ValueError("Word2Vec file SHA does not match current Project 3 formal corpus; retrain")
    if expected_corpus_sha256 is not None and report["corpus_sha256"] != expected_corpus_sha256:
        raise ValueError("Word2Vec source header checksum differs from current 3,000-document corpus")
    expected_models = {name.removesuffix(".model") for name in EMBEDDING_MODELS.values()}
    if set(report.get("experiments", {})) != expected_models:
        raise ValueError("Incomplete Project 3 Word2Vec experiments")
    for filename in EMBEDDING_MODELS.values():
        path_model = EMBEDDING_DIR / filename
        expected = report["experiments"][filename.removesuffix(".model")].get("model_sha256")
        if not path_model.is_file() or not expected or sha256(path_model) != expected:
            raise ValueError(f"Word2Vec model artifact missing/stale: {filename}")
    return report


def load_trained_word2vec(model_label, folder=None):
    """Load one existing model. No train(), no save(), no external network."""
    if model_label not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model: {model_label}")
    base = Path(folder) if folder is not None else EMBEDDING_DIR
    path = base / EMBEDDING_MODELS[model_label]
    if not path.is_file():
        raise FileNotFoundError(f"Word2Vec model missing: {path}")
    # Lazy dependency: the five core Project 3 pages need not import gensim.
    from gensim.models import Word2Vec
    model = Word2Vec.load(str(path))
    if not hasattr(model, "wv") or not hasattr(model.wv, "key_to_index"):
        raise ValueError("Artifact is not a supported Gensim Word2Vec model")
    return model


def normalized_single_token(raw):
    token = str(raw or "").strip().casefold()
    if not token:
        raise ValueError("Enter a single Word2Vec vocabulary token")
    if any(c.isspace() for c in token):
        raise ValueError("Word2Vec looks up one token at a time, not phrases")
    return token


def _freq(wv, word):
    try:
        return int(wv.get_vecattr(word, "count"))
    except (AttributeError, KeyError, ValueError, TypeError):
        return None


def compare_embeddings(model, raw_term, raw_other="", top_k=5):
    """Return actual model cosine neighbors & optional pair cosine or OOV guidance.

    Does not combine Word2Vec similarities with TF-IDF term weights, and does
    not produce or modify the retrieval rankings from Project 3.
    """
    term = normalized_single_token(raw_term)
    other = normalized_single_token(raw_other) if str(raw_other or "").strip() else None
    if not isinstance(top_k, int) or not 1 <= top_k <= 20:
        raise ValueError("Top-K must be between 1 and 20")
    wv = model.wv
    vocabulary = wv.key_to_index
    result = {"term": term, "other": other, "term_exists": term in vocabulary,
              "other_exists": other in vocabulary if other else None,
              "term_frequency": _freq(wv, term) if term in vocabulary else None,
              "neighbors": [], "pair_cosine": None, "suggestions": {}}
    if term in vocabulary:
        for rank, (neighbor, cosine) in enumerate(wv.most_similar(term, topn=top_k), 1):
            result["neighbors"].append({
                "Rank": rank, "Term": neighbor, "Cosine Similarity": float(cosine),
                "Training Frequency": _freq(wv, neighbor),
            })
    else:
        result["suggestions"][term] = get_close_matches(term, vocabulary, n=5, cutoff=0.55)
    if other and other not in vocabulary:
        result["suggestions"][other] = get_close_matches(other, vocabulary, n=5, cutoff=0.55)
    if other and term in vocabulary and other in vocabulary:
        result["pair_cosine"] = float(wv.similarity(term, other))
    return result
