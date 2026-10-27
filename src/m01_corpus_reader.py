"""M01: Load and validate the formal Project 3 3,000-document JATS abstract corpus.

Input is independent of downstream tokenization, weighting and retrieval.
The external Project3/data folder containing original JATS XML is NOT needed
at runtime; it stays in place for provenance and later projects.
"""
import gzip
import hashlib
import json
from pathlib import Path
from project3_config import CORPUS_PATH


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_corpus(path=None):
    """Return (documents, source header) or reject malformed corpus input."""
    path = Path(path) if path is not None else CORPUS_PATH
    docs, header = [], None
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_type") == "project2_corpus_header":
                if header is not None:
                    raise ValueError("Duplicate corpus headers")
                header = record
                continue
            abstract = str(record.get("abstract_clean") or record.get("abstract") or "").strip()
            if not abstract:
                raise ValueError(f"Empty abstract on line {line_number}, PMID={record.get('pmid')}")
            docs.append({
                "pmid": str(record.get("pmid") or ""),
                "pmcid": str(record.get("pmcid") or ""),
                "title": str(record.get("title") or "Untitled"),
                "abstract": abstract,
                "journal": str(record.get("journal") or ""),
                "doi": str(record.get("doi") or ""),
            })
    if not docs or len({d["pmid"] for d in docs}) != len(docs):
        raise ValueError("Corpus empty or duplicate PMID identifiers")
    if header is None:
        raise ValueError("Expected Project 2 corpus header")
    return docs, header
