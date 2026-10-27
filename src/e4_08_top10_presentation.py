"""Project 3 / 02 Top 10 presentation-only layout data.

Uses the five real retrieval outputs from A1; NEVER computes or changes rankings.
"""
from collections import Counter
import math

from e4_03_term_weighting import WEIGHTINGS


def build_five_top10_view(rankings, markers):
    """Return 5 per-method rows, a repeated Top10 PMID set, and CSV data.

    Repeated means present in >=2 DIFFERENT weighting-method Top10 lists.
    Marker is assigned by the original A1 pool and never re-numbered here.
    """
    missing = set(WEIGHTINGS) - set(rankings)
    if missing:
        raise ValueError(f"Missing weighting rankings: {sorted(missing)}")
    counts = Counter()
    for method in WEIGHTINGS:
        hits = rankings[method]["hits"]
        if len(hits) > 10:
            raise ValueError("Only Top 10 may be shown")
        ids = [str(hit["pmid"]) for hit in hits]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate PMID inside a single Top 10")
        counts.update(set(ids))
    repeated = frozenset(pmid for pmid, count in counts.items() if count >= 2)
    tables = {}
    csv_rows = []
    for method in WEIGHTINGS:
        rows = []
        for position, hit in enumerate(rankings[method]["hits"], 1):
            pmid = str(hit["pmid"])
            if pmid not in markers:
                raise ValueError(f"Missing A1 D ID for PMID {pmid}")
            if int(hit["rank"]) != position:
                raise ValueError("Top 10 ranks must be in retrieval order")
            score = float(hit["score"])
            if not math.isfinite(score) or score <= 0:
                raise ValueError("Top 10 needs positive finite cosine scores")
            row = {"Rank":position, "D ID":str(markers[pmid]),
                   "PMID":pmid, "Title":str(hit["title"]),
                   "Cosine Score":score, "Repeated":pmid in repeated}
            rows.append(row)
            csv_rows.append({"Weighting":method, **row})
        tables[method] = rows
    return {"tables":tables, "repeated_pmids":repeated, "csv_rows":csv_rows}
