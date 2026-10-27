"""Project 3 Batch 4: independently evaluate validated 3,000-document corpus.

No modifications to Streamlit defaults, the frozen 1,000/3,000 corpora,
Project 3's existing `results/` root reports, or its ten proxy rules.
All metrics are exploratory **regex-derived proxy** metrics, NOT human qrels.

Run from the Project 3 root:
    python src/e1_05_evaluate_3000_corpus.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

from e1_04_verify_3000_index import (
    BASELINE, CORPUS3000, REPORT_DIR, audit_source, ensure,
)
from e4_03_term_weighting import WEIGHTINGS
from e7_01_retrieval_evaluation import evaluate_all, load_rules, make_qrels
from e7_02_weighting_experiments import compare_normalization
from m01_corpus_reader import sha256
from m08_controller import SearchIndex
from project3_artifacts import write_csv, write_json
from project3_config import ROOT, RULES_PATH

OUTPUT_DIR = ROOT / "results" / "corpus_3000" / "condition_C"
BATCH3_REPORT = REPORT_DIR / "batch3_index3000_report_C.json"
METRICS = ("Precision@10", "Recall@10", "MRR@10", "nDCG@10")
REQUIRED_OUTPUTS = (
    "per_query_metrics_3000.csv", "mean_metrics_3000.csv",
    "per_query_metrics_1000.csv", "mean_metrics_1000.csv",
    "mean_metric_differences.csv", "proxy_qrel_counts.csv",
    "proxy_qrels_3000.json", "manual_review_queue.csv",
    "batch4_evaluation_report.json",
)


def verify_batch3():
    """Use the user's verified Batch 2/3 provenance, not simulated corpus data."""
    source = audit_source()  # repeats hash and first 1,000 full-record checks
    ensure(BATCH3_REPORT.is_file(), f"Missing Batch 3 PASS report: {BATCH3_REPORT}")
    batch3 = json.loads(BATCH3_REPORT.read_text(encoding="utf-8"))
    ensure(batch3.get("status") == "PASS", "Batch 3 index validation was not PASS")
    ensure(batch3.get("condition") == "C", "Batch 3 report must be preprocessing C")
    ensure(batch3.get("summary", {}).get("documents") == 3000,
           "Batch 3 report did not validate 3,000 documents")
    ensure(batch3.get("summary", {}).get("unique_pmids") == 3000,
           "Batch 3 report did not validate unique PMIDs")
    ensure(batch3.get("summary", {}).get("corpus_sha256") == sha256(CORPUS3000),
           "Batch 3 indexed file differs from current 3,000 corpus")
    ensure(batch3.get("source", {}).get("baseline_file_sha256") == sha256(BASELINE),
           "Baseline differs from Batch 3 validation")
    ensure(batch3.get("source", {}).get("corpus_3000_file_sha256") == sha256(CORPUS3000),
           "3,000 corpus differs from Batch 3 validation")
    return {"batch3_report": str(BATCH3_REPORT),
            "batch3_report_sha256": sha256(BATCH3_REPORT),
            "batch2_provenance": source}


def check_metrics(rows, means, rules, qrels, n, k=10):
    """Check coverage, denominators, label invariance and bounded IR metrics."""
    ensure(len(rules) == 10, f"Expected exactly 10 predefined queries; got {len(rules)}")
    ensure(len(rows) == len(WEIGHTINGS) * len(rules), "Expected 50 per-query rows")
    ensure(len(means) == len(WEIGHTINGS), "Expected five method mean rows")
    rule_ids = {r["id"] for r in rules}
    ensure(rule_ids == set(qrels), "Missing or unexpected qrel query IDs")
    ensure(all(0 < len(qrels[rid]) <= n for rid in rule_ids),
           "Empty / invalid proxy relevant set; evaluation is undefined")
    seen = set()
    for r in rows:
        pair = (r["Query ID"], r["Weighting"])
        ensure(pair not in seen, f"Repeated method/query: {pair}")
        seen.add(pair)
        ensure(pair[0] in rule_ids and pair[1] in WEIGHTINGS,
               f"Unexpected method/query: {pair}")
        ensure(r["Relevant Count"] == len(qrels[pair[0]]),
               f"Proxy-qrel total varies by method: {pair}")
        ensure(0 <= r["Relevant Retrieved"] <= r["Retrieved"] <= k,
               f"Invalid top-{k} counts: {pair}")
        for m in METRICS:
            ensure(r[m] is not None and math.isfinite(r[m]) and -1e-12 <= r[m] <= 1+1e-12,
                   f"Undefined/non-finite/out-of-bounds {m}: {pair}")
        ensure(math.isclose(r["Precision@10"], r["Relevant Retrieved"] / k,
                            abs_tol=1e-12), f"P@10 denominator mismatch: {pair}")
        ensure(math.isclose(r["Recall@10"], r["Relevant Retrieved"] / len(qrels[pair[0]]),
                            abs_tol=1e-12), f"Recall@10 denominator mismatch: {pair}")
        if r["Relevant Retrieved"] == 0:
            ensure(r["MRR@10"] == 0.0 and r["nDCG@10"] == 0.0,
                   f"Nonzero RR/nDCG with no relevant retrieval: {pair}")
    for s in means:
        group = [r for r in rows if r["Weighting"] == s["Weighting"]]
        ensure(len(group) == len(rules), "Incorrect method query count")
        for metric in METRICS:
            expected = sum(r[metric] for r in group) / len(group)
            ensure(math.isclose(s[metric], expected, abs_tol=1e-12),
                   f"Incorrect mean {metric} for {s['Weighting']}")


def manual_queue(index, rules, qrels, pool_per_method=10):
    """Pooled top results from ALL five methods, not one favored weighting.

    This is a **review queue**, not an edited qrels file. Blank judgments
    are intentionally empty. Accept 1=Relevant, 0=Nonrelevant, ?=Ambiguous.
    """
    rows = []
    for rule in rules:
        rid = rule["id"]
        pool = {}
        for method in WEIGHTINGS:
            hits = index.search(rule["query"], method, "Cosine", pool_per_method)["hits"]
            for h in hits:
                pmid = h["pmid"]
                if pmid not in pool:
                    pool[pmid] = {"document": h, "ranks": {}}
                pool[pmid]["ranks"][method] = int(h["rank"])
        for item in sorted(pool.values(), key=lambda x: (min(x["ranks"].values()),
                                                      int(x["document"]["doc_index"]))):
            h = item["document"]
            ranks = item["ranks"]
            rows.append({
                "Query ID": rid,
                "Query": rule["query"],
                "Topic": rule.get("topic", ""),
                "PMID": h["pmid"],
                "PMCID": h["pmcid"],
                "Title": h["title"],
                "Corpus origin": "Original 1000" if h["doc_index"] < 1000 else "New 2000",
                "Proxy relevant (0/1)": int(h["doc_index"] in qrels[rid]),
                "Ranks across five methods": "; ".join(
                    f"{method}={ranks[method]}" for method in WEIGHTINGS if method in ranks),
                "Main abstract excerpt": h["abstract"][:450],
                "Human relevance (1/0/?)": "",
                "Reviewer rationale": "",
            })
    return rows


def run(out_dir=OUTPUT_DIR, pool_per_method=10, verify_prerequisites=True,
        overwrite=False):
    out_dir = Path(out_dir)
    ensure(1 <= pool_per_method <= 10, "--pool-top must be 1..10")
    # Never allow a caller to accidentally overwrite old top-level output files.
    ensure(out_dir.resolve() != (ROOT / "results").resolve(),
           "Output must be in a separate subdirectory, not original results root")
    ensure((ROOT / "results").resolve() in out_dir.resolve().parents,
           "Output must be under Project 3 results/ and separate from old reports")
    existing = [p for name in REQUIRED_OUTPUTS if (p := out_dir / name).exists()]
    ensure(not existing or overwrite,
           "Batch 4 outputs already exist; use --overwrite to rebuild only this isolated folder")
    provenance = verify_batch3() if verify_prerequisites else {
        "testing_note": "Offline test only; NOT a validated real-corpus PASS"
    }
    print("PASS: earlier 3,000-document corpus/index provenance"
          if verify_prerequisites else "TEST ONLY: provenance guards intentionally skipped", flush=True)

    rules = load_rules()
    ensure(len(rules) == 10, "Batch 4 keeps exactly the original 10 defined queries")
    print("Indexing 1,000 and 3,000 corpus independently with preprocessing C ...", flush=True)
    one = SearchIndex.build(corpus_path=BASELINE, condition="C")
    three = SearchIndex.build(corpus_path=CORPUS3000, condition="C")
    ensure(len(one.docs) == 1000 and len(three.docs) == 3000,
           "Corpus sizes incorrect")
    ensure([d["pmid"] for d in three.docs[:1000]] ==
           [d["pmid"] for d in one.docs], "Original 1,000 order or PMIDs not preserved")
    ensure(three.corpus_sha256 == sha256(CORPUS3000) and one.corpus_sha256 == sha256(BASELINE),
           "Index not built from inspected corpus paths")
    if verify_prerequisites:
        batch3_summary = json.loads(BATCH3_REPORT.read_text(encoding="utf-8"))["summary"]
        current_summary = three.summary()
        for field in ("documents", "unique_pmids", "vocabulary", "tokens",
                      "condition", "corpus_sha256"):
            ensure(current_summary[field] == batch3_summary[field],
                   f"New evaluation index differs from validated Batch 3: {field}")

    rows1, means1, qrels1 = evaluate_all(one, rules, k=10)
    rows3, means3, qrels3 = evaluate_all(three, rules, k=10)
    check_metrics(rows1, means1, rules, qrels1, n=1000)
    check_metrics(rows3, means3, rules, qrels3, n=3000)
    if verify_prerequisites:
        # A lightweight independent guard that the previous 1,000-query setup
        # (rules, preprocessing, scoring) has not silently drifted.
        baseline_ndcg = next(s["nDCG@10"] for s in means1
                             if s["Weighting"] == "Log TF-IDF")
        ensure(math.isclose(baseline_ndcg, 0.9394434793486648, abs_tol=1e-6),
               "Frozen original 1,000-document evaluation no longer reproduces")
    for rule in rules:
        rid = rule["id"]
        old_pmids = {one.docs[i]["pmid"] for i in qrels1[rid]}
        nested_pmids = {three.docs[i]["pmid"] for i in qrels3[rid] if i < 1000}
        ensure(old_pmids == nested_pmids,
               f"Original 1,000 proxy labels differ after expansion: {rid}")
    print("PASS: 10 fixed queries × 5 methods × 2 corpora; bounded metrics and qrels", flush=True)

    report3_mean = next((i for i in means3 if i["Weighting"] == "Log TF-IDF"), None)
    ensure(report3_mean is not None, "Missing Log TF-IDF average")
    delta_rows = []
    old_by_method = {r["Weighting"]: r for r in means1}
    for new in means3:
        old = old_by_method[new["Weighting"]]
        entry = {"Weighting": new["Weighting"]}
        for m in METRICS:
            entry[f"{m} (1000)"] = old[m]
            entry[f"{m} (3000)"] = new[m]
            entry[f"{m} (3000 minus 1000)"] = new[m] - old[m]
        delta_rows.append(entry)
    counts = []
    for rule in rules:
        rid = rule["id"]
        a, b = len(qrels1[rid]), len(qrels3[rid])
        counts.append({
            "Query ID": rid, "Query": rule["query"], "Topic": rule.get("topic", ""),
            "Proxy relevant (1000)": a,
            "Proxy relevant (new 2000)": b - a,
            "Proxy relevant (3000)": b,
            "Proxy relevant fraction (1000)": a / 1000,
            "Proxy relevant fraction (3000)": b / 3000,
            "Relevance source": "regex on main abstract, not manual judgment",
        })
    pooled = manual_queue(three, rules, qrels3, pool_per_method)
    ensure(all(any(r["Query ID"] == rule["id"] for r in pooled) for rule in rules),
           "No manual review candidates for at least one query")
    normalize = compare_normalization(three, "GLP-1 kidney diabetes", "Raw TF-IDF", 10)
    ensure(normalize["max_absolute_error"] < 1e-10,
           "Cosine and L2 dot values differ beyond tolerance")
    print("PASS: baseline/new relevance counts; pooled unjudged manual-review queue", flush=True)

    result = {
        "status": "PASS" if verify_prerequisites else "SYNTHETIC_TEST_ONLY",
        "batch": "4: 3000 proxy-qrels evaluation and matched 1000 comparison",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": "C",
        "corpus_1000_sha256": one.corpus_sha256,
        "corpus_3000_sha256": three.corpus_sha256,
        "relevance_rules_sha256": sha256(RULES_PATH),
        "relevance_type": "unverified regular-expression thematic proxy; not human/independent gold qrels",
        "limitations": [
            "The same main-abstract lexical evidence influences both retrieval and rule-based qrels (circularity).",
            "Regex topic presence does not establish clinical/scientific relevance or negation/context.",
            "Qrel recall denominator changes as 2,000 documents are added.",
            "Differences reflect the nested corpus, fixed rules and five weighting methods, NOT proven general superiority.",
            "The manual review queue contains NO completed human judgments.",
        ],
        "source": provenance,
        "summary_1000": one.summary(), "summary_3000": three.summary(),
        "query_count": len(rules), "method_count": len(WEIGHTINGS),
        "per_query_rows_each_corpus": len(rows3),
        "qrel_counts_by_query": {r["Query ID"]: {
            "1000": r["Proxy relevant (1000)"],
            "added_2000": r["Proxy relevant (new 2000)"],
            "3000": r["Proxy relevant (3000)"],
        } for r in counts},
        "mean_metrics_1000": means1, "mean_metrics_3000": means3,
        "mean_metric_differences": delta_rows,
        "unjudged_manual_review_queue_rows": len(pooled),
        "manual_pool_per_method": pool_per_method,
        "rq4_cosine_l2_error": normalize["max_absolute_error"],
        "output_dir": str(out_dir),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "per_query_metrics_3000.csv", rows3)
    write_csv(out_dir / "mean_metrics_3000.csv", means3)
    write_csv(out_dir / "per_query_metrics_1000.csv", rows1)
    write_csv(out_dir / "mean_metrics_1000.csv", means1)
    write_csv(out_dir / "mean_metric_differences.csv", delta_rows)
    write_csv(out_dir / "proxy_qrel_counts.csv", counts)
    write_json(out_dir / "proxy_qrels_3000.json", {
        rid: sorted(three.docs[i]["pmid"] for i in ids) for rid, ids in qrels3.items()
    })
    write_csv(out_dir / "manual_review_queue.csv", pooled)
    write_json(out_dir / "batch4_evaluation_report.json", result)
    print(json.dumps({
        "status": result["status"], "condition": "C",
        "corpus_documents": [1000, 3000], "queries": len(rules),
        "weightings": len(WEIGHTINGS), "evaluations_per_corpus": len(rows3),
        "manual_review_queue_rows": len(pooled),
        "log_tfidf_ndcg_at_10_1000": old_by_method["Log TF-IDF"]["nDCG@10"],
        "log_tfidf_ndcg_at_10_3000": report3_mean["nDCG@10"],
        "output": str(out_dir),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool-top", type=int, default=10,
                   help="Top k per method per query for human review queue (1..10)")
    p.add_argument("--overwrite", action="store_true",
                   help="Explicitly replace only the separate Batch 4 output files")
    args = p.parse_args(argv)
    try:
        run(pool_per_method=args.pool_top, overwrite=args.overwrite)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
