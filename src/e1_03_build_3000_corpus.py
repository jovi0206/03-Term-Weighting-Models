"""E1-03: Build a separate Project 3 3,000-abstract corpus (Batch 2).

Default layout (run from ``03-term weighting models``):
  python src/e1_03_build_3000_corpus.py

Requires successful Batch 1 audit in the external raw-data pool; never changes
original JATS files, original 1,000-document prepared corpus or M01-M09.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from e1_02_legacy_jats_parser import parse_jats_abstract_only
from m01_corpus_reader import read_corpus

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT.parent / "data" / "glp1_pubmed_jats_3300"
DEFAULT_BASELINE = ROOT / "prepared_data" / "glp1_1000_abstracts.jsonl.gz"
DEFAULT_OUTPUT = ROOT / "prepared_data" / "glp1_3000_abstracts.jsonl.gz"
FROZEN_1000_FILE_SHA256 = "f69df9c1a9228ab5f9b1442ced6ba59ce783ded2e599462bd7391fef640e0bb1"
REQUIRED_MANIFEST_COLUMNS = {"rank", "pmid", "pmcid", "file", "title", "abstract_chars", "language"}
REQUIRED_SELECTION_COLUMNS = REQUIRED_MANIFEST_COLUMNS | {"origin", "corpus_order"}
N_CANDIDATES, N_BASELINE, N_NEW, N_FORMAL = 3300, 1000, 2000, 3000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        return list(reader)


def load_baseline(path: Path) -> tuple[list[dict], dict, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    file_hash = digest(path)
    if file_hash != FROZEN_1000_FILE_SHA256:
        raise ValueError(f"Frozen baseline checksum differs: {file_hash}")
    header, docs = None, []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("record_type") == "project2_corpus_header":
                if header is not None or docs:
                    raise ValueError("Original corpus header absent, duplicated or out of order")
                header = obj
            else:
                docs.append(obj)
    pmids = [str(d.get("pmid") or "").strip() for d in docs]
    if header is None or len(docs) != N_BASELINE or any(not p for p in pmids) or len(set(pmids)) != N_BASELINE:
        raise ValueError("Expected validated original 1,000 distinct PMID records")
    if header.get("corpus_sha256") != document_checksum(docs):
        raise ValueError("Frozen baseline corpus content hash mismatch")
    return docs, header, file_hash


def document_checksum(documents: list[dict]) -> str:
    """Byte-equivalent implementation of Project 2 E12 corpus_sha256()."""
    hasher = hashlib.sha256()
    for d in documents:
        record = {"pmid": d.get("pmid", ""), "pmcid": d.get("pmcid", ""),
                  "title": d.get("title", ""), "abstract": d.get("abstract", "")}
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        hasher.update(payload + b"\n")
    return hasher.hexdigest()


def inspect_manifest_and_selection(pool: Path, old: list[dict], original_sha: str):
    manifest_path = pool / "valid_jats_manifest.csv"
    audit_path = pool / "reports" / "batch1_jats_audit.json"
    selection_path = pool / "reports" / "batch1_proposed_3000_selection.csv"
    for path in (manifest_path, audit_path, selection_path, pool / "raw_jats"):
        if not path.exists():
            raise FileNotFoundError(f"Missing Batch 1 prerequisite: {path}")
    with audit_path.open("r", encoding="utf-8") as f:
        audit = json.load(f)
    manifest_hash = digest(manifest_path)
    if (audit.get("mode") != "manifest_and_raw_xml"
            or audit.get("ready_for_batch2") is not True
            or audit.get("errors") != []
            or audit.get("candidate_manifest_rows") != N_CANDIDATES
            or audit.get("raw_xml_verified") != N_CANDIDATES
            or audit.get("baseline_unique_pmids") != N_BASELINE
            or audit.get("baseline_pmids_in_manifest") != N_BASELINE
            or audit.get("new_pmids_in_manifest") != 2300
            or audit.get("manifest_sha256") != manifest_hash
            or audit.get("original_corpus_sha256") != original_sha):
        raise ValueError("Batch 1 audit missing, not passing, or source checksums changed. Rerun Batch 1 audit")

    rows = load_csv(manifest_path, REQUIRED_MANIFEST_COLUMNS)
    if len(rows) != N_CANDIDATES:
        raise ValueError(f"Expected {N_CANDIDATES} candidate rows, got {len(rows)}")
    try:
        ordered = sorted(rows, key=lambda row: int(row["rank"]))
        ranks = [int(r["rank"]) for r in ordered]
    except (ValueError, TypeError) as ex:
        raise ValueError("Manifest ranks must be integer values") from ex
    if len(set(ranks)) != N_CANDIDATES:
        raise ValueError("Manifest ranks not distinct")
    pmids = [r["pmid"].strip() for r in ordered]
    pmcids = [r["pmcid"].strip() for r in ordered]
    if len(set(pmids)) != N_CANDIDATES or len(set(pmcids)) != N_CANDIDATES:
        raise ValueError("Manifest PMID or PMCID duplicated")
    if any(r["file"] != f'{r["pmcid"]}.xml' for r in ordered):
        raise ValueError("Manifest contains invalid PMC filename")
    lookup = {r["pmid"]: r for r in ordered}
    original_pmids = [str(d["pmid"]) for d in old]
    original_set = set(original_pmids)
    if not original_set.issubset(lookup):
        raise ValueError("Original corpus PMIDs not all present in Manifest")
    new_candidates = [r for r in ordered if r["pmid"] not in original_set]
    if len(new_candidates) != 2300:
        raise ValueError("Expected exactly 2300 new candidates")

    proposed = load_csv(selection_path, REQUIRED_SELECTION_COLUMNS)
    if len(proposed) != N_FORMAL:
        raise ValueError(f"Expected {N_FORMAL} selection records; got {len(proposed)}")
    selected_pmids = [r["pmid"] for r in proposed]
    expected_pmids = original_pmids + [r["pmid"] for r in new_candidates[:N_NEW]]
    if selected_pmids != expected_pmids:
        raise ValueError("Selection changed: expected exact frozen 1000 order + first 2000 new by rank")
    for i, row in enumerate(proposed):
        origin = "frozen_1000" if i < N_BASELINE else "new_jats"
        if row["origin"] != origin or row["corpus_order"] != str(i + 1):
            raise ValueError(f"Wrong selection order/origin on row {i + 1}")
        match = lookup[row["pmid"]]
        if any(row[col] != match[col] for col in REQUIRED_MANIFEST_COLUMNS):
            raise ValueError(f"Selection/Manifest mismatch on PMID={row['pmid']}")
    return proposed, audit, {
        "manifest_sha256": manifest_hash,
        "audit_sha256": digest(audit_path),
        "selection_sha256": digest(selection_path),
        "selected_new_rank_min": int(proposed[N_BASELINE]["rank"]),
        "selected_new_rank_max": int(proposed[-1]["rank"]),
    }


def parse_new_2000(pool: Path, selected: list[dict], legacy_schema: list[str]) -> list[dict]:
    new_documents: list[dict] = []
    for i, row in enumerate(selected[N_BASELINE:], N_BASELINE + 1):
        path = pool / "raw_jats" / row["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing selected new XML: {path}")
        doc = parse_jats_abstract_only(path)
        pmid, pmcid = (str(doc.get(x) or "").strip() for x in ("pmid", "pmcid"))
        if (pmid != row["pmid"] or pmcid != row["pmcid"]
                or doc.get("filename") != row["file"]
                or doc.get("title") != row["title"]
                or str(doc.get("language") or "").casefold() != row["language"].casefold()):
            raise ValueError(f"Original Project 2 parser disagrees with Manifest: {path.name}")
        abstract = doc.get("abstract", "")
        if (not abstract.strip() or len(abstract) != int(row["abstract_chars"])
                or doc.get("abstract_clean") != abstract
                or doc.get("statistics_text") != abstract
                or doc.get("body") != "" or doc.get("body_clean") != ""):
            raise ValueError(f"Main Abstract/body scope differs from expected: {path.name}")
        if set(doc) != set(legacy_schema):
            raise ValueError(f"Legacy document fields differ for PMID={pmid}: {sorted(set(doc) ^ set(legacy_schema))}")
        new_documents.append(doc)
        if len(new_documents) % 250 == 0:
            print(f"[PARSE] {len(new_documents):,}/{N_NEW:,} new Project 2-compatible JATS", flush=True)
    return new_documents


def write_corpus_atomic(path: Path, header: dict, docs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing 3,000 corpus: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=".glp1_3000_tmp_", suffix=".jsonl.gz", dir=path.parent)
    tmp = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as zipped:
                for item in [header, *docs]:
                    zipped.write((json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8"))
        loaded_docs, loaded_header = read_corpus(tmp)
        if len(loaded_docs) != N_FORMAL or loaded_header.get("corpus_sha256") != header["corpus_sha256"]:
            raise ValueError("Project 3 M01 corpus contract failed for new data")
        if len({d["pmid"] for d in loaded_docs}) != N_FORMAL:
            raise ValueError("Project 3 M01 found duplicate new PMIDs")
        with gzip.open(tmp, "rt", encoding="utf-8") as check:
            raw_header = json.loads(next(check))
            roundtrip_docs = [json.loads(line) for line in check if line.strip()]
        if (raw_header != header or len(roundtrip_docs) != N_FORMAL
                or document_checksum(roundtrip_docs) != header["corpus_sha256"]
                or roundtrip_docs != docs):
            raise ValueError("Full JSONL.gzip round-trip verification failed")
        if path.exists():
            raise FileExistsError(f"Output appeared during verification; refusing overwrite: {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def build(pool: Path, baseline: Path, output: Path) -> dict:
    pool, baseline, output = (Path(p).resolve() for p in (pool, baseline, output))
    if baseline == output:
        raise ValueError("Refusing to overwrite the original 1,000-document corpus")
    if output.is_file():
        raise FileExistsError(f"Output already exists, will not overwrite: {output}")
    frozen_docs, frozen_header, frozen_file_sha = load_baseline(baseline)
    selection, audit, hashes = inspect_manifest_and_selection(pool, frozen_docs, frozen_file_sha)
    print(f"Verified Batch 1 audit and frozen baseline; first {N_BASELINE:,} records preserved", flush=True)
    added_docs = parse_new_2000(pool, selection, list(frozen_docs[0]))
    docs = frozen_docs + added_docs
    all_pmids = [str(d["pmid"]) for d in docs]
    if len(docs) != N_FORMAL or len(set(all_pmids)) != N_FORMAL:
        raise ValueError("3,000 distinct PMID corpus not achieved")
    if any(d.get("source_format") != "JATS" or not str(d.get("abstract") or "").strip() for d in docs):
        raise ValueError("A record has invalid source format or missing abstract")
    corpus_content_sha = document_checksum(docs)
    new_audit = {
        "target_documents": N_FORMAL,
        "accepted_documents": len(docs),
        "unique_pmids": len(set(all_pmids)),
        "source_format": "PMC JATS XML -> Project 2 E12 main abstract only",
        "search_scope": "abstract_only",
        "analysis_scope": "main_abstract_body_only",
        "abstract_policy": "Project 2 E12 unchanged: untyped main > first non-excluded; paragraph text only; titles excluded",
        "selection_policy": "Preserve frozen 1,000 records in original order, append first 2,000 new distinct PMIDs by manifest relevance rank",
        "source_candidate_documents": N_CANDIDATES,
        "original_1000_reused_unchanged": N_BASELINE,
        "new_2000_parsed_with_legacy": N_NEW,
        "provenance": {
            "baseline_prepared_corpus_sha256": frozen_file_sha,
            "baseline_header_corpus_sha256": frozen_header["corpus_sha256"],
            **hashes,
            "raw_xml_pool": str(pool / "raw_jats"),
            "source_batch1_audit_ready": audit["ready_for_batch2"],
        },
    }
    header = {"record_type": "project2_corpus_header", "schema_version": 1,
              "corpus_sha256": corpus_content_sha, "audit": new_audit}
    write_corpus_atomic(output, header, docs)
    report = {
        "batch": "2: Build formal 3,000-document prepared corpus",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "original_1000_preserved": N_BASELINE,
        "new_2000_project2_parsed": N_NEW,
        "formal_documents": len(docs),
        "unique_pmids": len(set(all_pmids)),
        "schema_fields": list(docs[0]),
        "all_body_fields_empty": all(d.get("body") == "" and d.get("body_clean") == "" for d in docs),
        "corpus_content_sha256": corpus_content_sha,
        "corpus_file_sha256": digest(output),
        "corpus_path": str(output),
        "baseline_corpus_sha256_unchanged": digest(baseline) == frozen_file_sha,
        "baseline_file_sha256": frozen_file_sha,
        "source_batch1_audit_sha256": hashes["audit_sha256"],
        "source_manifest_sha256": hashes["manifest_sha256"],
        "source_selection_sha256": hashes["selection_sha256"],
        "warnings": audit.get("warnings", []),
        "next_step": "Batch 3: reindex / re-evaluate separately; project3_config.py remains on 1,000 baseline",
    }
    report_path = pool / "reports" / "batch2_corpus3000_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in (
        "status", "original_1000_preserved", "new_2000_project2_parsed",
        "formal_documents", "unique_pmids", "all_body_fields_empty",
        "corpus_content_sha256", "corpus_file_sha256", "baseline_corpus_sha256_unchanged"
    )}, indent=2, ensure_ascii=False))
    print(f"Corpus: {output}")
    print(f"Report: {report_path}")
    print("Project 3 defaults and original 1,000 corpus have NOT changed.")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)
    try:
        build(args.pool, args.baseline, args.output)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
