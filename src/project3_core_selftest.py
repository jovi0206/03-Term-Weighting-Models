"""Run required Project 3 math, corpus, ranking and evaluation checks without pytest.

Usage: python src/project3_core_selftest.py
"""
import math
from m08_controller import SearchIndex
from e4_03_term_weighting import tf_weight, WEIGHTINGS
from e7_01_retrieval_evaluation import evaluate_all, load_rules
from e7_02_weighting_experiments import compare_normalization
from m02_text_processor import tokens, idf
from m01_corpus_reader import sha256
from project3_config import CORPUS_PATH, FORMAL_CORPUS_FILE_SHA256


def main():
    assert tokens("The GLP-1 and insulin", "C") == ["glp-1", "insulin"]
    assert idf(3000, 300) == 1.0
    assert tf_weight(10, "Log TF", 0.5) == 2.0
    index = SearchIndex.build()
    assert sha256(CORPUS_PATH) == FORMAL_CORPUS_FILE_SHA256
    assert index.summary()["documents"] == 3000
    assert index.summary()["unique_pmids"] == 3000
    assert len(index.vocabulary) > 20000
    rules = load_rules()
    rows, means, qrels = evaluate_all(index, rules)
    assert len(rules) == 10 and len(rows) == 50 and len(means) == len(WEIGHTINGS)
    assert all(qrels.values())
    for method in WEIGHTINGS:
        test = compare_normalization(index, "GLP-1 kidney diabetes", method)
        assert test["cosine_l2_same_order"]
        assert test["max_absolute_error"] < 1e-10
    assert index.search("thisworddoesnotexist")["hits"] == []
    assert index.search("thisworddoesnotexist")["oov"] == ["thisworddoesnotexist"]
    print("PASS: 3,000 corpus records; 5 weights; 10 proxy queries; 50 evaluations; cosine/L2; OOV")
    print("SUMMARY:", index.summary())
    for method in means:
        print(method["Weighting"], "nDCG@10", f"{method['nDCG@10']:.6f}")


if __name__ == "__main__":
    main()
