"""Batch 2 regression: shared weighting, vector parity, chart invariance, UI contract.

No independent TF/IDF formula is introduced: indexes, query vectors and
E4's original policies are the numerical sources of truth.
"""
from collections import Counter
from pathlib import Path
import ast
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from e4_03_term_weighting import WEIGHTINGS, tf_weight
from e4_04_two_term_comparison import compare_two_terms
from e4_05_weighting_comparison import compare_query_document_terms
from m04_query_processor import query_vector
from m02_text_processor import tokens
from m08_controller import SearchIndex
from project3_charts import (
    build_single_term_five_weight_chart, build_query_document_comparison_chart,
    build_two_term_five_weight_chart,
)


@pytest.fixture(scope="module")
def ix():
    return SearchIndex.build(condition="C")


def test_five_methods_in_same_corpus_vocabulary_and_original_vectors(ix):
    pmid = "35503498"
    doc_idx = next(i for i, d in enumerate(ix.docs) if d["pmid"] == pmid)
    query = "glp-1 semaglutide semaglutide"
    term_rows, weighted_rows = compare_query_document_terms(ix, query, doc_idx, ["glp-1", "semaglutide"])
    assert len(term_rows) == 2
    assert len(weighted_rows) == 10
    assert [row["Weighting"] for row in weighted_rows] == list(WEIGHTINGS)*2

    originals = compare_two_terms(ix, "glp-1", "semaglutide", doc_idx)
    vectors = {m: query_vector(ix, query, m)[0] for m in WEIGHTINGS}
    for summary, original in zip(term_rows, originals):
        term = summary["Term"]
        col = ix.vocabulary[term]
        qtf = Counter(tokens(query, ix.condition))[term]
        idf = ix.idfs[col]
        assert summary["Query TF"] == qtf
        assert summary["Document TF"] == original["Doc TF"]
        assert summary["Corpus DF"] == original["Corpus DF"]
        assert summary["IDF (log10)"] == pytest.approx(float(idf))
        for method in WEIGHTINGS:
            item = next(row for row in weighted_rows if row["Term"] == term and row["Weighting"] == method)
            assert item["Query Weight"] == pytest.approx(float(vectors[method][0, col]))
            assert item["Document Weight"] == pytest.approx(float(ix.matrices[method][doc_idx, col]))
            assert item["Document Weight"] == pytest.approx(original[method])
            assert item["Query Weight"] == pytest.approx(tf_weight(qtf, method, idf))


def test_single_term_query_zero_weight_and_oov(ix):
    doc_idx = next(i for i, d in enumerate(ix.docs) if d["pmid"] == "35503498")
    summary, rows = compare_query_document_terms(ix, "glp-1", doc_idx, ["semaglutide"])
    assert len(summary) == 1 and len(rows) == 5
    assert summary[0]["Query TF"] == 0
    assert all(row["Query Weight"] == 0 for row in rows)
    assert any(row["Document Weight"] > 0 for row in rows)

    summary, rows = compare_query_document_terms(ix, "", doc_idx, ["unknownmedicalwordxyz"])
    assert summary[0]["Corpus DF"] == 0 and summary[0]["IDF (log10)"] is None
    assert all(row["Query Weight"] == row["Document Weight"] == 0 for row in rows)


def test_invalid_and_duplicate_tokens_handled(ix):
    with pytest.raises(ValueError, match="same term"):
        compare_query_document_terms(ix, "glp-1", 0, ["GLP-1", "glp-1"])
    with pytest.raises(ValueError, match="One token"):
        compare_query_document_terms(ix, "", 0, ["semaglutide weight"])
    with pytest.raises(ValueError, match="one or two"):
        compare_query_document_terms(ix, "", 0, [])
    with pytest.raises(IndexError):
        compare_query_document_terms(ix, "", 1000, ["glp-1"])


def test_all_preprocessing_variants_supported(ix):
    for condition in "ABCD":
        idx = ix if condition == "C" else SearchIndex.build(condition=condition)
        summaries, rows = compare_query_document_terms(idx, "glp-1", 0, ["glp-1"])
        assert len(summaries) == 1 and len(rows) == 5
        for row in rows:
            assert row["Weighting"] in WEIGHTINGS
            assert np.isfinite(row["Document Weight"])


def test_charts_show_unmodified_weight_values(ix):
    doc_idx = next(i for i, d in enumerate(ix.docs) if d["pmid"] == "35503498")
    _, weights = compare_query_document_terms(ix, "glp-1 semaglutide", doc_idx,
                                              ["glp-1", "semaglutide"])
    first = weights[:5]
    fig1 = build_single_term_five_weight_chart(first)
    fig2 = build_query_document_comparison_chart(first)
    fig3 = build_two_term_five_weight_chart(weights)
    assert len(fig1.data) == len(fig2.data) == len(fig3.data) == 5
    for i, method in enumerate(WEIGHTINGS):
        assert fig1.data[i].y[0] == pytest.approx(first[i]["Document Weight"])
        assert list(fig2.data[i].y) == pytest.approx([first[i]["Query Weight"], first[i]["Document Weight"]])
        assert list(fig3.data[i].y) == pytest.approx([
            next(row["Document Weight"] for row in weights if row["Term"]==term and row["Weighting"]==method)
            for term in ["glp-1", "semaglutide"]
        ])
    assert fig1.layout.yaxis.range[0] == 0
    assert fig2.layout.yaxis2.range[0] == 0
    with pytest.raises(ValueError):
        build_single_term_five_weight_chart(first[:2])
    with pytest.raises(ValueError):
        build_two_term_five_weight_chart(first)


def test_new_ui_location_without_core_rewrites():
    root = Path(__file__).resolve().parents[1]
    app = (root / "src" / "m09_app.py").read_text(encoding="utf-8")
    ast.parse(app)
    assert app.index('"01 Retrieval"') < app.index('"01-1 Weighting Comparison"') < app.index('"02 Term Weighting"')
    assert 'elif page == "01-1 Weighting Comparison"' in app
    assert "compare_query_document_terms(" in app
    assert "build_query_document_comparison_chart(" in app
    assert "Word2Vec comparison " in app  # Deliberately deferred until Batch 3
