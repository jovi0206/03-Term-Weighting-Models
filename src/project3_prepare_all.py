"""Prepare all required Project 3 offline evidence.

Run: python src/project3_prepare_all.py --condition C
Output: formal C -> results/; exploratory A/B/D -> results/conditions/{A,B,D}
"""
import argparse
import json
import platform
from datetime import datetime, timezone

from e7_01_retrieval_evaluation import evaluate_all, load_rules
from e7_02_weighting_experiments import compare_normalization
from m01_corpus_reader import sha256
from m08_controller import SearchIndex
from project3_artifacts import write_csv, write_json
from project3_config import (ROOT, RULES_PATH, RESULTS_PATH, CORPUS_PATH,
                             FORMAL_DOCUMENT_COUNT, FORMAL_CORPUS_FILE_SHA256,
                             CONDITIONS, FORMAL_CONDITION, DEFAULT_QUERY)
from e4_03_term_weighting import WEIGHTINGS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, default=FORMAL_CONDITION)
    args = parser.parse_args(argv)
    index = SearchIndex.build(condition=args.condition)
    if len(index.docs) != FORMAL_DOCUMENT_COUNT or index.corpus_sha256 != FORMAL_CORPUS_FILE_SHA256:
        raise RuntimeError("Wrong formal 3,000-document corpus; results will not be overwritten")
    rules = load_rules()
    rows, summary, qrels = evaluate_all(index, rules)
    output = RESULTS_PATH if args.condition == FORMAL_CONDITION else RESULTS_PATH / "conditions" / args.condition
    output.mkdir(parents=True, exist_ok=True)

    write_csv(output / "per_query_metrics.csv", rows)
    write_csv(output / "mean_metrics.csv", summary)
    write_csv(output / "demo_top10_logtfidf_cosine.csv",
              index.search(DEFAULT_QUERY, "Log TF-IDF", "Cosine", 10)["hits"])
    comparison = compare_normalization(index, DEFAULT_QUERY)
    variants = comparison["hits"]
    write_csv(output / "demo_rq4_normalization.csv", [
        {
            "Rank": rank + 1,
            **{f"{metric} PMID": hits[rank]["pmid"] if len(hits) > rank else ""
               for metric, hits in variants.items()},
            **{f"{metric} Score": hits[rank]["score"] if len(hits) > rank else None
               for metric, hits in variants.items()},
        }
        for rank in range(10)
    ])

    lineage = ["e2_01_porter.py", "e2_02_preprocessing_experiment.py", "e4_02_term_statistics.py"]
    project3 = [p.name for p in sorted((ROOT / "src").glob("*.py")) if p.name not in lineage]
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "course_assignment": "Project #3: Term Weighting / Vector Space / Retrieval Evaluation",
        "source": "Project 3 formal 3,000 GLP-1 abstracts: 1,000 original + 2,000 additional Project 2-compatible PMC JATS Main Abstracts",
        "source_sha256_file": index.corpus_sha256,
        "runtime_python": platform.python_version(),
        "legacy_sha256": {name: sha256(ROOT / "src" / name) for name in lineage},
        "project3_sha256": {name: sha256(ROOT / "src" / name) for name in project3},
        "source_project2_header_sha256": index.header.get("corpus_sha256"),
        "relevance_rules_sha256": sha256(RULES_PATH),
        "judgments": "deterministic thematic regex proxy, not independent/manual ground truth",
        "preprocessing": args.condition,
        "idf": "log10(N/DF), unsmoothed; df=N -> 0",
        "log_tf": "1 + log10(tf) for tf>0; else 0",
        "query_document_weighting": "symmetric (same selected formula); binary tf for IDF only",
        "ranking": "cosine; positive scores only; tie -> corpus source order",
        "zero_relevance": "undefined metrics, excluded from mean; this run has none",
        "precision": "P@10 always divides by 10 even if fewer positive-score hits",
        "summary": index.summary(), "num_queries": len(rules),
        "qrel_counts": {rule["id"]: len(qrels[rule["id"]]) for rule in rules},
        "method_order": WEIGHTINGS, "mean_metrics": summary,
        "rq4_cosine_l2_same_order": comparison["cosine_l2_same_order"],
        "output_dir": str(output.relative_to(ROOT)),
    }
    write_json(output / "reproducibility_manifest.json", manifest)
    write_json(output / "proxy_qrels.json", {
        rid: sorted(index.docs[i]["pmid"] for i in ids) for rid, ids in qrels.items()
    })
    print(json.dumps({"corpus": index.summary(), "qrel_counts": manifest["qrel_counts"],
                      "methods": summary, "output_dir": str(output)}, indent=2, ensure_ascii=False))
    return index, rows, summary, qrels


if __name__ == "__main__":
    main()
