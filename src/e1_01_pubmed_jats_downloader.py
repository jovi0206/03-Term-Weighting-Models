"""Project 3 E1: download PubMed-ranked PMC JATS candidate XMLs.

Standalone stdlib downloader; does NOT alter the frozen 1,000-document corpus.
The main-abstract screening policy mirrors Project 2 Extension E12.

Run from the Project 3 root:
    python src/e1_01_pubmed_jats_downloader.py --email you@school.edu --target 20
    python src/e1_01_pubmed_jats_downloader.py --email you@school.edu --target 3300

Resumes using an append-only audit log. Network access required on your PC.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

PUBMED_QUERY = '("GLP-1"[All Fields] OR "glucagon-like peptide-1"[All Fields])'
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
IDCONV = "https://pmc.ncbi.nlm.nih.gov/tools/idconv/api/v1/articles/"
TOOL = "ncku_glp1_jats_project3"
EXCLUDED_ABSTRACT_TYPE_MARKERS = (
    "author-highlights", "highlight", "highlights", "precis", "graphical",
    "teaser", "plain-language", "plain language", "key-points", "key points",
    "publisher", "summary-for", "supplement",
)
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"


def tag_text(node: ET.Element | None) -> str:
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def article_id(meta: ET.Element, id_type: str) -> str:
    for node in meta.findall("./{*}article-id"):
        if (node.get("pub-id-type") or "").casefold() == id_type:
            return tag_text(node)
    return ""


def inspect_jats(blob: bytes, expected_pmid: str, expected_pmcid: str):
    """Check the same essential acceptance rules used by Project 2 E12.

    Returns a single <article> XML and brief audit details. Does not inspect body.
    Final 3,000-document corpus must still be rebuilt with the original full loader.
    """
    root = ET.fromstring(blob)
    article = root if root.tag.split("}")[-1] == "article" else root.find(".//{*}article")
    if article is None:
        raise ValueError("response does not contain JATS <article>")
    front = article.find("./{*}front")
    meta = front.find("./{*}article-meta") if front is not None else None
    if meta is None:
        raise ValueError("missing front/article-meta")
    pmid = article_id(meta, "pmid")
    pmcid = article_id(meta, "pmcid") or article_id(meta, "pmc")
    if not pmid or not pmcid:
        raise ValueError("missing PMID or PMCID")
    if pmid != expected_pmid or pmcid != expected_pmcid:
        raise ValueError(f"ID mismatch: XML has PMID={pmid} PMCID={pmcid}")
    language = (article.get(XML_LANG) or article.get("xml:lang") or "").strip().casefold()
    if language and not language.startswith("en"):
        raise ValueError(f"non-English xml:lang={language}")
    abstracts = meta.findall("./{*}abstract")
    usable = []
    for node in abstracts:
        abstract_type = (node.get("abstract-type") or node.get("content-type") or "").strip().casefold()
        if not any(marker in abstract_type for marker in EXCLUDED_ABSTRACT_TYPE_MARKERS):
            usable.append(node)
    untyped = [node for node in usable if not (node.get("abstract-type") or node.get("content-type") or "").strip()]
    main = (untyped or usable or [None])[0]
    if main is None:
        raise ValueError("no main abstract (only excluded types or absent)")
    sections = main.findall("./{*}sec")
    if sections:
        paragraphs = []
        for sec in sections:
            ps = [tag_text(p) for p in sec.findall(".//{*}p") if tag_text(p)]
            if not ps:
                label = tag_text(sec.find("./{*}title"))
                t = tag_text(sec)
                if label and t.startswith(label):
                    t = t[len(label):].strip()
                if t:
                    ps = [t]
            paragraphs.extend(ps)
    else:
        paragraphs = [tag_text(p) for p in main.findall(".//{*}p") if tag_text(p)]
        if not paragraphs and tag_text(main):
            paragraphs = [tag_text(main)]
    abstract = " ".join(paragraphs).strip()
    if not abstract:
        raise ValueError("empty selected main abstract")
    title = tag_text(meta.find("./{*}title-group/{*}article-title"))
    # JATS from PMC EFetch can be wrapped by <pmc-articleset>. Store <article> root
    # to match the original user's PMCxxxxx.xml files.
    data = blob if root is article else ET.tostring(article, encoding="utf-8", xml_declaration=True)
    return data, {"pmid": pmid, "pmcid": pmcid, "title": title, "abstract_chars": len(abstract), "language": language}


class Client:
    def __init__(self, email: str, pause: float, key: str = ""):
        self.email = email
        self.pause = max(0.55, pause)  # well below 3 requests/second, including conversion API
        self.key = key
        self.last_request = 0.0

    def get(self, base: str, params: dict, retries: int = 5) -> bytes:
        p = dict(params)
        p.setdefault("tool", TOOL)
        p.setdefault("email", self.email)
        if self.key and base.startswith(EUTILS):
            p.setdefault("api_key", self.key)
        url = base + "?" + urllib.parse.urlencode(p)
        headers = {"User-Agent": f"{TOOL}/1.0 ({self.email})", "Accept": "application/xml,application/json"}
        for attempt in range(retries):
            dt = time.monotonic() - self.last_request
            if dt < self.pause:
                time.sleep(self.pause - dt)
            self.last_request = time.monotonic()
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                    data = r.read()
                    if not data:
                        raise ValueError("empty HTTP body")
                    return data
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"Request failed after {retries} attempts: {type(exc).__name__}: {exc}") from exc
                time.sleep(min(32, 2 ** attempt + 1))
        raise AssertionError("unreachable")


def search_pubmed(client: Client, max_pmids: int) -> tuple[list[str], int]:
    raw = client.get(EUTILS + "/esearch.fcgi", {
        "db": "pubmed", "term": PUBMED_QUERY, "retmax": max_pmids,
        "retstart": 0, "sort": "relevance", "retmode": "xml",
    })
    root = ET.fromstring(raw)
    if root.find("ERROR") is not None:
        raise RuntimeError(tag_text(root.find("ERROR")))
    count = int(root.findtext("Count") or "0")
    pmids = [n.text.strip() for n in root.findall("./IdList/Id") if n.text]
    if not pmids:
        raise RuntimeError("No PubMed PMIDs returned; inspect query/network")
    return pmids, count


def convert_pmids(client: Client, pmids: list[str]) -> dict[str, str]:
    raw = client.get(IDCONV, {"ids": ",".join(pmids), "idtype": "pmid", "format": "json"})
    data = json.loads(raw)
    if data.get("status") != "ok":
        raise RuntimeError(f"PMC ID conversion error: {str(data)[:400]}")
    result = {}
    for record in data.get("records", []):
        pmid = str(record.get("pmid") or record.get("requested-id") or "")
        pmcid = str(record.get("pmcid") or "")
        if pmid in pmids and re.fullmatch(r"PMC\d+", pmcid):
            result[pmid] = pmcid
    return result


def fetch_pmc_jats(client: Client, pmcid: str) -> bytes:
    return client.get(EUTILS + "/efetch.fcgi", {
        "db": "pmc", "id": pmcid.removeprefix("PMC"), "retmode": "xml",
    })


def load_events(path: Path, jats_dir: Path) -> dict[str, dict]:
    state: dict[str, dict] = {}
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    obj = json.loads(line)
                    state[str(obj["pmid"])] = obj
    # An accepted file missing on disk is never counted as successful.
    for pmid in list(state):
        record = state[pmid]
        if record.get("status") == "accepted" and not (jats_dir / f"{record['pmcid']}.xml").is_file():
            del state[pmid]
    return state


def append_event(log: Path, obj: dict) -> None:
    obj = dict(obj)
    obj["time_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with log.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def write_manifest(folder: Path, state: dict) -> None:
    rows = sorted(
        (r for r in state.values() if r.get("status") == "accepted"),
        key=lambda r: int(r.get("rank", 999999)),
    )
    path = folder / "valid_jats_manifest.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["rank", "pmid", "pmcid", "file", "title", "abstract_chars", "language"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})
    summary = {
        "target": None, "accepted": len(rows),
        "rejected_or_not_in_pmc": sum(r.get("status") != "accepted" for r in state.values()),
        "search_query": PUBMED_QUERY,
        "search_sort": "PubMed relevance", "main_abstract_policy": "Project2 E12 compatible",
        "xml_folder": str(folder / "raw_jats"),
        "warning": "JATS validated here; final 3,000 corpus requires original loader audit and new evaluation",
    }
    (folder / "download_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def download(args: argparse.Namespace) -> int:
    out = Path(args.output).expanduser().resolve() if args.output else Path(__file__).resolve().parents[2] / "data" / "glp1_pubmed_jats_3300"
    jats_dir = out / "raw_jats"
    jats_dir.mkdir(parents=True, exist_ok=True)
    log = out / "download_events.jsonl"
    state = load_events(log, jats_dir)
    client = Client(args.email, args.pause, os.getenv("NCBI_API_KEY", ""))
    snapshot = out / "pubmed_pmids_ranked.txt"
    if snapshot.exists():
        pmids = [line.strip() for line in snapshot.read_text(encoding="utf-8").splitlines() if line.strip()]
        previous_meta_path = out / "search_metadata.json"
        previous_meta = json.loads(previous_meta_path.read_text(encoding="utf-8")) if previous_meta_path.exists() else {}
        pubmed_total = previous_meta.get("pubmed_match_count", len(pmids))
        print("Reusing FROZEN PMID relevance order from previous run.")
    else:
        pmids, pubmed_total = search_pubmed(client, args.max_pmids)
        snapshot.write_text("\n".join(pmids) + "\n", encoding="utf-8")
        (out / "search_metadata.json").write_text(json.dumps({
            "query": PUBMED_QUERY, "sort": "relevance", "pubmed_match_count": pubmed_total,
            "pmids_considered": len(pmids), "download_target_initial": args.target,
            "source": "PubMed ESearch -> PMC ID Converter -> PMC EFetch JATS",
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    accepted = sum(r.get("status") == "accepted" for r in state.values())
    print(f"PubMed matches: {pubmed_total:,}; examined first {len(pmids):,} by relevance")
    print(f"Resume: {accepted:,} accepted XML files already logged in {jats_dir}")
    print(f"Target: {args.target:,} valid JATS, output: {out}")
    if accepted >= args.target:
        write_manifest(out, state)
        print("Target already complete; nothing downloaded.")
        return 0
    for start in range(0, len(pmids), 200):
        if accepted >= args.target:
            break
        group = pmids[start:start + 200]
        pending = [p for p in group if p not in state or state[p].get("status") == "fetch_error" or (args.retry_failed and state[p].get("status") != "accepted")]
        if not pending:
            continue
        mapped = convert_pmids(client, pending)
        for offset, pmid in enumerate(group):
            if accepted >= args.target:
                break
            if pmid not in pending:
                continue
            rank = start + offset + 1
            pmcid = mapped.get(pmid)
            if not pmcid:
                rec = {"rank": rank, "pmid": pmid, "status": "no_pmc_mapping"}
            else:
                path = jats_dir / f"{pmcid}.xml"
                try:
                    if path.is_file():
                        data, audit = inspect_jats(path.read_bytes(), pmid, pmcid)
                    else:
                        data, audit = inspect_jats(fetch_pmc_jats(client, pmcid), pmid, pmcid)
                        tmp = path.with_suffix(".xml.part")
                        tmp.write_bytes(data)
                        tmp.replace(path)
                    rec = {"rank": rank, "pmid": pmid, "pmcid": pmcid,
                           "file": path.name, "status": "accepted", **audit}
                    accepted += 1
                except Exception as exc:
                    rec = {"rank": rank, "pmid": pmid, "pmcid": pmcid,
                           "status": "fetch_error" if isinstance(exc, RuntimeError) else "rejected",
                           "reason": f"{type(exc).__name__}: {exc}"}
            state[pmid] = rec
            append_event(log, rec)
            if accepted % 25 == 0 and rec["status"] == "accepted":
                write_manifest(out, state)
                print(f"[OK] {accepted:,}/{args.target:,} JATS; PubMed rank {rank:,}; latest {pmcid}", flush=True)
    write_manifest(out, state)
    if accepted < args.target:
        print(f"INCOMPLETE: only {accepted:,}/{args.target:,} valid JATS within {len(pmids):,} PubMed hits.")
        print("Keep files; extend candidate search strategy after checking rejection report. Do not count as 3,300.")
        return 2
    print(f"DONE: {accepted:,} validated XML files; manifest: {out / 'valid_jats_manifest.csv'}")
    print("Original 1,000-document prepared corpus has NOT been modified.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch 3,300 GLP-1 PMC JATS articles in PubMed relevance order.")
    parser.add_argument("--email", default=os.getenv("NCBI_EMAIL", ""), help="Real NCBI contact email (or NCBI_EMAIL env var)")
    parser.add_argument("--target", type=int, default=3300, help="Target valid JATS XML count")
    parser.add_argument("--max-pmids", type=int, default=10000, help="How many PubMed-ranked PMIDs to inspect (<=10000)")
    parser.add_argument("--pause", type=float, default=0.65, help="Minimum seconds between requests; always >=0.55")
    parser.add_argument("--output", help="Override default external Project3/data/glp1_pubmed_jats_3300")
    parser.add_argument("--retry-failed", action="store_true", help="Revisit rejected/no-map IDs (normally skip them)")
    args = parser.parse_args()
    if not args.email or "@" not in args.email:
        parser.error("--email must be a real contact email; you can also set NCBI_EMAIL")
    if not 1 <= args.target <= 10000:
        parser.error("--target must be between 1 and 10000")
    if not 1 <= args.max_pmids <= 10000:
        parser.error("--max-pmids must be between 1 and 10000")
    return download(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Downloaded files and audit log remain; rerun same command to resume.")
        sys.exit(130)
