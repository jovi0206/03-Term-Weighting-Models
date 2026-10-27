"""Professor's week-two additions: reproducible term comparison and interpretation."""
import gzip
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from m08_controller import SearchIndex
from e4_04_two_term_comparison import compare_two_terms
from e7_03_weighting_interpretation import method_performance_summary
from project3_charts import build_two_term_weight_chart
from e4_03_term_weighting import WEIGHTINGS


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    folder = tmp_path_factory.mktemp("p3_teacher")
    file = folder / "mini.jsonl.gz"
    abstracts = ["alpha alpha alpha beta", "alpha beta gamma",
                 "beta gamma gamma", "gamma delta"]
    with gzip.open(file, "wt", encoding="utf-8") as output:
        output.write(json.dumps({"record_type": "project2_corpus_header", "corpus_sha256": "mini"}) + "\n")
        for num, abstract in enumerate(abstracts, 1):
            output.write(json.dumps({"pmid": str(num), "title": f"Doc {num}",
                                     "abstract_clean": abstract}) + "\n")
    return SearchIndex.build(file, condition="B")


def test_two_terms_share_corpus_idf_but_keep_doc_tf(small_index):
    rows = compare_two_terms(small_index, "alpha", "beta", 0)
    alpha, beta = rows
    assert [r["Term"] for r in rows] == ["alpha", "beta"]
    assert (alpha["Doc TF"], beta["Doc TF"]) == (3, 1)
    assert alpha["Corpus DF"] == 2 and beta["Corpus DF"] == 3
    assert alpha["IDF (log10)"] == pytest.approx(math.log10(4 / 2))
    assert beta["IDF (log10)"] == pytest.approx(math.log10(4 / 3))
    assert alpha["Raw TF-IDF"] == pytest.approx(3 * math.log10(2))
    assert alpha["Log TF-IDF"] == pytest.approx((1 + math.log10(3)) * math.log10(2))
    assert beta["Log TF"] == pytest.approx(1.0)


def test_two_terms_oov_and_absent_in_document(small_index):
    rows = compare_two_terms(small_index, "alpha", "doesnotexist", 2)
    assert rows[0]["Doc TF"] == 0 and rows[0]["Raw TF-IDF"] == 0
    assert rows[0]["IDF (log10)"] is not None
    assert rows[1]["Corpus DF"] == 0 and rows[1]["IDF (log10)"] is None
    assert all(rows[1][method] == 0 for method in WEIGHTINGS)


def test_two_terms_invalid_inputs_are_not_silently_split(small_index):
    with pytest.raises(ValueError, match="exactly one"):
        compare_two_terms(small_index, "alpha gamma", "beta", 0)
    with pytest.raises(ValueError, match="same token"):
        compare_two_terms(small_index, "Alpha", "alpha", 0)
    with pytest.raises(IndexError):
        compare_two_terms(small_index, "alpha", "beta", 100)


def test_two_term_chart_shows_doc_weights_only(small_index):
    rows = compare_two_terms(small_index, "alpha", "beta", 0)
    figure = build_two_term_weight_chart(rows)
    assert figure.layout.barmode == "group"
    assert len(figure.data) == 2
    assert [float(v) for v in figure.data[0].y] == pytest.approx(
        [r["Raw TF-IDF"] for r in rows]
    )


def test_ranks_are_experiment_specific_and_ties_share_rank():
    vals = [.50, .70, .80, .80, .90]
    means = [{"Weighting": m, "Precision@10": .6, "nDCG@10": v}
             for m, v in zip(WEIGHTINGS, vals)]
    evidence = method_performance_summary(means)
    assert [item["nDCG rank (of 5)"] for item in evidence] == [5, 4, 2, 2, 1]
    with pytest.raises(ValueError):
        method_performance_summary(means[:-1])


def test_teacher_ux_sections_wired_to_pure_helpers():
    source = (ROOT / "src" / "m09_app.py").read_text(encoding="utf-8")
    assert "Two-term comparison · teacher follow-up" in source
    assert "Teacher Task 4 — experimental control" in source
    assert "Why can TF-IDF be useful without always winning?" in source
    assert "compare_two_terms(index,term_a,term_b,pmid)" in source
    assert "method_performance_summary(means)" in source
