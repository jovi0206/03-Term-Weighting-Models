"""E1-04 / Batch 3: verify the separate Project 3 3,000-document VSM index.

Run from ``03-term weighting models``:
    python src/e1_04_verify_3000_index.py

Read-only for both prepared corpora, XML pool, SRC, existing results and UI.
The sole written file is a new validation report under the external raw-data
pool's reports folder. This does NOT switch the Streamlit default corpus and
DOES NOT produce/relabel proxy relevance judgments or evaluation metrics.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from e4_03_term_weighting import WEIGHTINGS, tf_weight
from m01_corpus_reader import read_corpus, sha256
from m02_text_processor import idf
from m05_similarity import score_documents
from m04_query_processor import query_vector
from m08_controller import SearchIndex
from project3_artifacts import write_json
from project3_config import CONDITIONS, ROOT

BASELINE = ROOT / "prepared_data" / "glp1_1000_abstracts.jsonl.gz"
CORPUS3000 = ROOT / "prepared_data" / "glp1_3000_abstracts.jsonl.gz"
REPORT_DIR = ROOT.parent / "data" / "glp1_pubmed_jats_3300" / "reports"
BATCH2_REPORT = REPORT_DIR / "batch2_corpus3000_report.json"

# Expected from the user's actual Batch 2 PASS output, not from a synthetic test.
BASELINE_SHA256 = "f69df9c1a9228ab5f9b1442ced6ba59ce783ded2e599462bd7391fef640e0bb1"
CORPUS3000_SHA256 = "c67cef4af612bab3a32ca6b1c13229a74a965a4e29d823a7118c37dcb171ede6"
CONTENT3000_SHA256 = "4ab676e5de41cdbb661fceb1714b075834dce9ee30e61aa3ad59df4e5ca8f4f5"


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def iter_jsonl(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def audit_source(baseline_path=BASELINE, corpus_path=CORPUS3000,
                 report_path=BATCH2_REPORT, require_frozen_hashes=True):
    """Check file identities and exact, complete first-1,000 record equality."""
    baseline_path, corpus_path, report_path = map(Path, (baseline_path, corpus_path, report_path))
    for path in (baseline_path, corpus_path, report_path):
        ensure(path.is_file(), f"Missing prerequisite: {path}")

    original_file_hash, expanded_file_hash = sha256(baseline_path), sha256(corpus_path)
    with report_path.open("r", encoding="utf-8") as f:
        batch2 = json.load(f)
    ensure(batch2.get("status") == "PASS", "Batch 2 report not PASS")
    ensure(batch2.get("formal_documents") == 3000 and batch2.get("unique_pmids") == 3000,
           "Batch 2 count or PMID unique check differs")
    ensure(batch2.get("all_body_fields_empty") is True,
           "Batch 2 body exclusion check missing")
    ensure(batch2.get("baseline_corpus_sha256_unchanged") is True,
           "Batch 2 did not verify frozen original baseline")
    ensure(batch2.get("baseline_file_sha256") == original_file_hash,
           "Baseline hash differs from the one audited by Batch 2")
    ensure(batch2.get("corpus_file_sha256") == expanded_file_hash,
           "3,000 corpus file changed since Batch 2")
    if require_frozen_hashes:
        ensure(original_file_hash == BASELINE_SHA256,
               "Original 1,000 corpus differs from frozen validated Project 3")
        ensure(expanded_file_hash == CORPUS3000_SHA256,
               "3,000 corpus differs from the user's Batch 2 PASS output")
        ensure(batch2.get("corpus_content_sha256") == CONTENT3000_SHA256,
               "3,000 corpus content hash differs from Batch 2 PASS output")

    original_objects, expanded_objects = list(iter_jsonl(baseline_path)), list(iter_jsonl(corpus_path))
    ensure(len(original_objects) == 1001 and len(expanded_objects) == 3001,
           "Expected one header plus exactly 1,000 / 3,000 complete records")
    original_header, original_records = original_objects[0], original_objects[1:]
    expanded_header, expanded_records = expanded_objects[0], expanded_objects[1:]
    ensure(original_header.get("record_type") == "project2_corpus_header" and
           expanded_header.get("record_type") == "project2_corpus_header",
           "Corpus header record type differs")
    ensure(expanded_header.get("corpus_sha256") == batch2["corpus_content_sha256"],
           "Expanded corpus header checksum differs from Batch 2 report")
    ensure(expanded_records[:1000] == original_records,
           "First 1,000 complete JSON records are NOT identical to the baseline")
    # Project 2's semantic content digest: independent of gzip bytes and JSONL layout.
    hasher = hashlib.sha256()
    for record in expanded_records:
        semantic_fields = {key: record.get(key, "") for key in
                           ("pmid", "pmcid", "title", "abstract")}
        payload = json.dumps(semantic_fields, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        hasher.update(payload + b"\n")
    ensure(hasher.hexdigest() == expanded_header["corpus_sha256"],
           "Actual 3,000 records do not match Project 2 semantic content SHA256")
    pmids = [str(r.get("pmid", "")).strip() for r in expanded_records]
    ensure(all(pmids) and len(set(pmids)) == 3000, "Expanded corpus PMID check failed")
    ensure(all(r.get("body") == "" and r.get("body_clean") == "" for r in expanded_records),
           "Expanded corpus includes nonempty Body")
    ensure(all((r.get("abstract_clean") or "").strip() for r in expanded_records),
           "Expanded corpus includes empty abstract_clean")
    ensure(len(original_records[0]) == 42 and
           all(set(r) == set(original_records[0]) for r in expanded_records),
           "Expanded corpus's 42-field schema differs from original Project 2 schema")
    return {
        "baseline_file_sha256": original_file_hash,
        "corpus_3000_file_sha256": expanded_file_hash,
        "corpus_3000_content_sha256": expanded_header["corpus_sha256"],
        "baseline_records_identical": 1000,
        "expanded_records": 3000,
        "schema_field_count": len(original_records[0]),
        "batch2_report": str(report_path),
        "batch2_report_sha256": sha256(report_path),
    }


def audit_index(index: SearchIndex) -> dict:
    """Validate TF/CF/DF, IDF, five sparse matrices and their L2 norms."""
    n, vocab_size = len(index.docs), len(index.vocabulary)
    ensure(n == 3000, f"Wrong indexed document count: {n}")
    ensure(len({d["pmid"] for d in index.docs}) == 3000, "Indexed PMID not unique")
    ensure(vocab_size > 0 and len(index.counters) == n, "Vocabulary or document TF missing")
    ensure(set(index.df) == set(index.cf) == set(index.vocabulary), "Term inventories differ")
    ensure(list(index.vocabulary.values()) == list(range(vocab_size)),
           "Vocabulary column order is not contiguous")

    reconstructed_cf = Counter()
    reconstructed_df = Counter()
    for counter in index.counters:
        reconstructed_cf.update(counter)
        reconstructed_df.update(counter.keys())
    ensure(reconstructed_cf == index.cf, "CF differs from original term counts")
    ensure(reconstructed_df == index.df, "DF differs from number of containing documents")
    ensure(sum(index.cf.values()) == sum(map(lambda x: sum(x.values()), index.counters)),
           "Corpus token total differs from TF total")
    expected_idfs = np.array([idf(n, index.df[term]) for term in index.vocabulary], dtype=float)
    ensure(np.all(np.isfinite(index.idfs)), "IDF has nonfinite values")
    ensure(np.allclose(index.idfs, expected_idfs, atol=1e-12, rtol=0),
           "IDF differs from log10(N/DF)")
    ensure(np.all(index.idfs >= 0), "Negative IDF is unexpected in unsmoothed log10(N/DF)")
    ensure(set(index.matrices) == set(WEIGHTINGS) == set(index.norms),
           "Expected exactly five weighting matrices with norms")

    matrix_report = {}
    sampled_row_ids = sorted(set([0, 1, 999, 1000, n - 1]))
    for method in WEIGHTINGS:
        matrix, norms = index.matrices[method], index.norms[method]
        ensure(matrix.shape == (n, vocab_size), f"{method}: wrong matrix shape")
        ensure(norms.shape == (n,), f"{method}: wrong L2 norm shape")
        ensure(matrix.nnz > 0 and np.all(np.isfinite(matrix.data)),
               f"{method}: empty matrix or nonfinite values")
        ensure(np.all(matrix.data >= 0) and np.all(np.isfinite(norms)) and np.all(norms >= 0),
               f"{method}: unexpected negative/nonfinite weight or norm")
        actual_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
        ensure(np.allclose(actual_norms, norms, atol=1e-10, rtol=1e-12),
               f"{method}: indexed norms do not match weighted vectors")
        for row in sampled_row_ids:
            for term, tf in list(index.counters[row].items())[:20]:
                col = index.vocabulary[term]
                expected = tf_weight(tf, method, float(index.idfs[col]))
                actual = float(matrix[row, col])
                ensure(math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12),
                       f"{method}: wrong weight for row={row}, term={term}")
        matrix_report[method] = {
            "rows": n, "columns": vocab_size, "nonzero_weights": int(matrix.nnz),
            "zero_norm_documents": int(np.count_nonzero(norms == 0)),
        }
    return {"summary": index.summary(), "weight_matrices": matrix_report}


def audit_retrieval(index: SearchIndex) -> dict:
    """Sample top-10 ranking and raw full-vector Cosine vs L2 agreement."""
    query = "semaglutide weight loss"
    more_query = "glp-1 kidney diabetes"
    checks = {}
    for method in WEIGHTINGS:
        hits = index.search(query, method=method, metric="Cosine", limit=10)["hits"]
        ensure(len(hits) >= 1, f"{method}: no retrieval results for {query!r}")
        ensure(all(0 < h["score"] <= 1.0 + 1e-10 for h in hits),
               f"{method}: cosine scores out of expected range")
        ensure(all(a["score"] >= b["score"] for a, b in zip(hits, hits[1:])),
               f"{method}: Top 10 scores not sorted")
        for sample_query in (query, more_query):
            q, oov, _ = query_vector(index, sample_query, method)
            cosine, _ = score_documents(index, q, method, "Cosine")
            explicit_l2, _ = score_documents(index, q, method, "L2 Dot Product")
            max_error = float(np.max(np.abs(cosine - explicit_l2)))
            ensure(max_error < 1e-10, f"{method}: Cosine != L2 dot for {sample_query!r}")
            if sample_query == query:
                l2_hits = index.search(query, method=method,
                                       metric="L2 Dot Product", limit=10)["hits"]
                order_identical = [h["pmid"] for h in hits] == [h["pmid"] for h in l2_hits]
                checks[method] = {
                    "query": query,
                    "top10_pmids": [h["pmid"] for h in hits],
                    "top10_cosine_l2_same_order": order_identical,
                    "max_full_vector_cosine_l2_error": max_error,
                }
    unknown = "nonexistentbiomedicalterm987654321xyz"
    oov = index.search(unknown, "Raw TF", "Cosine", 10)
    ensure(not oov["hits"] and oov["oov"] == [unknown], "OOV query not handled correctly")
    return {"tested_queries": [query, more_query, unknown],
            "top10_by_weighting": checks,
            "oov_query_returns_no_hits": True}


def run(condition="C", output=None):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown preprocessing: {condition}")
    source = audit_source()
    print("PASS: Batch 2 corpus provenance, exact baseline 1,000, 3,000 unique PMIDs", flush=True)
    print(f"Indexing 3,000 abstracts with preprocessing {condition} ...", flush=True)
    index = SearchIndex.build(corpus_path=CORPUS3000, condition=condition)
    ensure(index.corpus_sha256 == source["corpus_3000_file_sha256"],
           "M01/M03 indexed a different file from source audit")
    computed = audit_index(index)
    print("PASS: TF / CF / DF / log10 IDF / five sparse vectors / L2 norms", flush=True)
    retrieval = audit_retrieval(index)
    print("PASS: sample Top 10, 5 weightings, Cosine = normalized L2 dot, OOV", flush=True)
    sample_terms = {}
    for term in ("glp-1", "semaglutide", "insulin", "kidney", "diabetes"):
        stats = index.term_stats(term)
        sample_terms[term] = stats
    result = {
        "status": "PASS", "batch": "3: index and retrieval validation (no proxy qrels yet)",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "condition": condition, "source": source, **computed, "retrieval": retrieval,
        "sample_term_statistics": sample_terms,
        "default_app_corpus_untouched": True,
        "original_result_files_untouched": True,
        "next_step": "Batch 4: regenerate 10-query proxy judgments and IR metrics in separate corpus_3000 result directory; recheck relevance-judgment limitations",
    }
    destination = Path(output) if output else REPORT_DIR / f"batch3_index3000_report_{condition}.json"
    write_json(destination, result)
    print(json.dumps({
        "status": result["status"], "documents": index.summary()["documents"],
        "unique_pmids": index.summary()["unique_pmids"],
        "vocabulary": index.summary()["vocabulary"],
        "tokens": index.summary()["tokens"],
        "corpus_file_sha256": index.corpus_sha256,
        "methods": list(WEIGHTINGS), "report": str(destination),
    }, ensure_ascii=False, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=CONDITIONS, default="C",
                        help="preprocessing condition; C is the formal fixed baseline")
    parser.add_argument("--output", type=Path, default=None,
                        help="optional separate JSON validation report location")
    args = parser.parse_args(argv)
    try:
        run(args.condition, args.output)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
