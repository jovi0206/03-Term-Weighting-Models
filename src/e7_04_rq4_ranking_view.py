"""RQ4 display data derived from unchanged M08/M06 retrieval results.

No weighting, similarity, normalization, scoring, tokenization, or ranking is
implemented here. All three searches call the same index.search facade.
This module only aligns PMIDs, identifies cross-method repeats and prepares
export/detail rows for the 03 Controlled Experiments user interface.
"""
from __future__ import annotations

from math import isfinite
from collections import Counter

RQ4_METRICS = (
    ("Raw Dot Product", "Raw Dot"),
    ("Cosine", "Cosine"),
    ("L2 Dot Product", "L2 Dot"),
)
RQ4_METRIC_KEYS = tuple(method for method, _ in RQ4_METRICS)


def make_rq4_view(index, query: str, weighting: str = "Raw TF-IDF", k: int = 10):
    """Return presentation-only RQ4 comparison, using actual retrieval scores.

    Top-3 presentation colors use a PMID seen in at least two of the three
    ranking lists (including Cosine/L2 repeats). This is visual identification,
    not evidence from independent retrieval models. The Top-10 overlap metric
    and Top-10 table highlighting keep their original Raw Dot vs Cosine scope.
    Cosine-vs-L2 score error is matched by *PMID*, never arbitrary row number.
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    rankings = {}
    for metric in RQ4_METRIC_KEYS:
        result = index.search(query, weighting, metric, k)
        hits = list(result["hits"])
        pmids = [str(hit["pmid"]) for hit in hits]
        if len(set(pmids)) != len(pmids):
            raise ValueError(f"Repeated PMID within one {metric} ranking")
        for hit in hits:
            if not isfinite(float(hit["score"])):
                raise ValueError(f"Non-finite retrieval score: {metric}")
        rankings[metric] = result

    raw = rankings["Raw Dot Product"]["hits"]
    cosine = rankings["Cosine"]["hits"]
    l2_dot = rankings["L2 Dot Product"]["hits"]
    # For the aligned Top-3 cards, a PMID is repeated if it appears in >=2
    # distinct scoring-mode columns. This does NOT change ranking or stats.
    top3_counts = Counter(
        str(hit["pmid"])
        for metric in RQ4_METRIC_KEYS
        for hit in rankings[metric]["hits"][:3]
    )
    top3_repeat = {pmid for pmid, count in top3_counts.items() if count >= 2}
    top10_repeat = {str(h["pmid"]) for h in raw} & {str(h["pmid"]) for h in cosine}

    # One stable D marker per unique PMID, no matter the metric/rank.
    marker_by_pmid = {}
    hit_by_pmid = {}
    for metric in RQ4_METRIC_KEYS:
        for hit in rankings[metric]["hits"]:
            pmid = str(hit["pmid"])
            if pmid not in marker_by_pmid:
                marker_by_pmid[pmid] = f"D{len(marker_by_pmid)+1:02d}"
                hit_by_pmid[pmid] = hit

    cosine_scores = {str(hit["pmid"]): float(hit["score"]) for hit in cosine}
    l2_scores = {str(hit["pmid"]): float(hit["score"]) for hit in l2_dot}
    matching_pmids = cosine_scores.keys() & l2_scores.keys()
    same_order = bool(cosine) and bool(l2_dot) and (
        [str(h["pmid"]) for h in cosine] == [str(h["pmid"]) for h in l2_dot])
    score_error = (max(abs(cosine_scores[p] - l2_scores[p]) for p in matching_pmids)
                   if matching_pmids else None)

    tables = {}
    export_rows = []
    for metric, short_name in RQ4_METRICS:
        rows = []
        for hit in rankings[metric]["hits"]:
            pmid = str(hit["pmid"])
            row = {
                "Method": short_name,
                "Rank": int(hit["rank"]),
                "D ID": marker_by_pmid[pmid],
                "PMID": pmid,
                "Title": str(hit.get("title", "")),
                "Score": float(hit["score"]),
                "Raw-Cos Top 10 repeat": pmid in top10_repeat,
            }
            rows.append(row)
            export_rows.append(row)
        tables[metric] = rows

    return {
        "rankings": rankings,
        "markers": marker_by_pmid,
        "hit_by_pmid": hit_by_pmid,
        "top3_repeated_pmids": top3_repeat,
        "top10_repeated_pmids": top10_repeat,
        "raw_cos_top10_overlap": len(top10_repeat),
        "cos_l2_same_order": same_order,
        "max_cos_l2_error": score_error,
        "cos_l2_shared_pmids": len(matching_pmids),
        "tables": tables,
        "export_rows": export_rows,
        "no_positive_hits": not any(rankings[m]["hits"] for m in RQ4_METRIC_KEYS),
        "oov": sorted({term for result in rankings.values() for term in result.get("oov", [])}),
    }
