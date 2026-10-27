"""E1 (Project 3): read-only JATS 3300 audit and proposed 3000 selection.

Does not change downloaded XML, the 1000-document corpus or Project 3 code.
Uses E1-01's JATS validator; this is NOT the formal Project2 corpus builder.

Run from the Project3 project directory, with your normal venv:
    python src/e1_02_jats_corpus_audit.py

For metadata-only preflight (does not inspect XML bytes):
    python src/e1_02_jats_corpus_audit.py --manifest-only
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from e1_01_pubmed_jats_downloader import inspect_jats

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT.parent / "data" / "glp1_pubmed_jats_3300"
DEFAULT_CORPUS = ROOT / "prepared_data" / "glp1_1000_abstracts.jsonl.gz"
REQUIRED = {"rank", "pmid", "pmcid", "file", "title", "abstract_chars", "language"}
FROZEN_1000_SHA256 = "f69df9c1a9228ab5f9b1442ced6ba59ce783ded2e599462bd7391fef640e0bb1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def baseline(path: Path) -> tuple[list[dict], dict]:
    documents, header = [], None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "project2_corpus_header":
                if header is not None:
                    raise ValueError("More than one Project 2 header")
                header = record
            else:
                documents.append(record)
    if header is None or len(documents) != 1000:
        raise ValueError("Expected a frozen 1,000-document Project 2 corpus and its header")
    pmids = [str(x.get("pmid") or "").strip() for x in documents]
    if any(not x for x in pmids) or len(set(pmids)) != 1000:
        raise ValueError("Baseline PMIDs must contain 1,000 unique nonempty values")
    return documents, header


def manifest_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"Manifest missing fields: {sorted(REQUIRED - set(reader.fieldnames or []))}")
        rows = list(reader)
    return rows


def audit(manifest: Path, xml_dir: Path, corpus: Path, output: Path, manifest_only: bool) -> int:
    if not manifest.is_file() or not corpus.is_file():
        raise FileNotFoundError(f"Missing manifest or baseline corpus: {manifest}; {corpus}")
    rows = manifest_rows(manifest)
    old, header = baseline(corpus)
    current_baseline_sha256 = sha256(corpus)
    old_pmids = [str(d["pmid"]).strip() for d in old]
    old_set = set(old_pmids)
    errors, warnings = [], []
    if current_baseline_sha256 != FROZEN_1000_SHA256:
        errors.append("Frozen 1,000-document corpus SHA-256 differs from validated Project 3 baseline")
    if len(rows) != 3300:
        errors.append(f"Expected exactly 3,300 manifest entries; found {len(rows)}")
    for field in ("pmid", "pmcid", "file", "rank"):
        values = [r.get(field, "").strip() for r in rows]
        missing = sum(not v for v in values)
        duplicates = len(values) - len(set(values))
        if missing or duplicates:
            errors.append(f"{field}: missing={missing}, duplicate={duplicates}")
    bad_fields = []
    for idx, row in enumerate(rows, 1):
        pmid, pmcid, filename = (row.get(x, "").strip() for x in ("pmid", "pmcid", "file"))
        rank, chars = row.get("rank", "").strip(), row.get("abstract_chars", "").strip()
        if (not pmid.isdecimal() or not re.fullmatch(r"PMC\d+", pmcid)
                or filename != f"{pmcid}.xml" or row.get("language", "").strip().casefold() != "en"
                or not rank.isdecimal() or not chars.isdecimal() or int(chars or 0) < 1):
            bad_fields.append(idx)
    if bad_fields:
        errors.append(f"Invalid manifest values in {len(bad_fields)} rows (first lines {bad_fields[:10]})")
    try:
        ordered = sorted(rows, key=lambda r: int(r["rank"]))
    except ValueError:
        ordered = rows
    present = {r.get("pmid", "").strip() for r in rows}
    missing_old = old_set - present
    if missing_old:
        errors.append(f"{len(missing_old)} frozen baseline PMIDs missing from candidate pool; examples {sorted(missing_old)[:10]}")
    new_rows = [r for r in ordered if r.get("pmid", "").strip() not in old_set]
    if len(new_rows) < 2000:
        errors.append(f"Need 2,000 new candidates, only {len(new_rows)} available")
    first_3000 = {r.get("pmid", "").strip() for r in ordered[:3000]}
    if len(first_3000 & old_set) != 1000:
        warnings.append(f"Only {len(first_3000 & old_set)} of baseline 1,000 are in first 3,000 ranks; selection will still preserve baseline")
    selected = []
    lookup = {r.get("pmid", "").strip(): r for r in rows}
    for rownum, pmid in enumerate(old_pmids, 1):
        row = lookup.get(pmid)
        if row:
            selected.append({"corpus_order": rownum, "origin": "frozen_1000", **row})
    for row in new_rows[:2000]:
        selected.append({"corpus_order": len(selected) + 1, "origin": "new_jats", **row})
    if len(selected) != 3000 or len({r["pmid"] for r in selected}) != 3000:
        errors.append(f"Proposed selection requires 3,000 unique PMIDs; found {len(selected)}")
    duplicate_titles = {k: v for k, v in Counter(r.get("title", "") for r in rows).items() if v > 1}
    if duplicate_titles:
        warnings.append(f"{len(duplicate_titles)} repeated title texts (PMIDs/PMCIDs distinct); manual review optional")
    xml_verified, xml_problems = 0, []
    if not manifest_only:
        if not xml_dir.is_dir():
            errors.append(f"Missing XML directory: {xml_dir}")
        else:
            for r in rows:
                pmid, pmcid = r["pmid"].strip(), r["pmcid"].strip()
                path = xml_dir / r["file"].strip()
                if not path.is_file():
                    xml_problems.append({"pmid": pmid, "pmcid": pmcid, "error": "missing XML file"})
                    continue
                try:
                    _, detail = inspect_jats(path.read_bytes(), pmid, pmcid)
                    if detail["abstract_chars"] != int(r["abstract_chars"]):
                        raise ValueError("Main Abstract character count differs from Manifest")
                    if detail["title"] != r["title"]:
                        raise ValueError("Article title differs from Manifest")
                    xml_verified += 1
                except (ValueError, OSError, KeyError, SyntaxError, TypeError) as e:
                    xml_problems.append({"pmid": pmid, "pmcid": pmcid, "error": f"{type(e).__name__}: {e}"})
            if xml_problems:
                errors.append(f"XML missing / invalid / changed: {len(xml_problems)} files")

    summary = {
        "run_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "manifest_only" if manifest_only else "manifest_and_raw_xml",
        "batch": "Batch 1 audit; this does NOT create or replace a formal corpus",
        "manifest": str(manifest), "manifest_sha256": sha256(manifest),
        "original_corpus": str(corpus), "original_corpus_sha256": current_baseline_sha256,
        "project2_source_header_sha256": header.get("corpus_sha256"),
        "candidate_manifest_rows": len(rows),
        "candidate_unique_pmids": len(present),
        "baseline_unique_pmids": len(old_set),
        "baseline_pmids_in_manifest": len(old_set & present),
        "new_pmids_in_manifest": len(present - old_set),
        "baseline_pmids_in_top3000_relevance_rows": len(old_set & first_3000),
        "proposed_formal_3000": len(selected),
        "proposed_new": len([r for r in selected if r["origin"] == "new_jats"]),
        "raw_xml_verified": xml_verified if not manifest_only else None,
        "raw_xml_errors": xml_problems[:100],
        "title_duplicates": duplicate_titles,
        "warnings": warnings, "errors": errors,
        "ready_for_batch2": bool(not errors and not manifest_only and xml_verified == len(rows)),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "batch1_jats_audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(selected) == 3000 and not errors:
        with (output / "batch1_proposed_3000_selection.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
    print(json.dumps({k: summary[k] for k in (
        "mode", "candidate_manifest_rows", "candidate_unique_pmids", "baseline_unique_pmids",
        "baseline_pmids_in_manifest", "new_pmids_in_manifest", "baseline_pmids_in_top3000_relevance_rows",
        "proposed_formal_3000", "raw_xml_verified", "ready_for_batch2", "warnings", "errors"
    )}, indent=2, ensure_ascii=False))
    print(f"Audit file: {output / 'batch1_jats_audit.json'}")
    return 0 if not errors else 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_POOL / "valid_jats_manifest.csv")
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_POOL / "raw_jats")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_POOL / "reports")
    parser.add_argument("--manifest-only", action="store_true", help="Check metadata only; cannot authorize formal 3000 build")
    opts = parser.parse_args(argv)
    return audit(opts.manifest, opts.xml_dir, opts.corpus, opts.output, opts.manifest_only)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"AUDIT FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(2)
