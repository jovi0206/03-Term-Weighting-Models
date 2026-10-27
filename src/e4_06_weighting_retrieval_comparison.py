"""E4: Five-method, one-query retrieval comparison for the learning UI (A1).

Only delegates to M08/SearchIndex.search; no second tokenization, scoring,
weighting, or ranking implementation. Pure result shaping for the UI and tests.
"""
from __future__ import annotations

from itertools import combinations
from collections import Counter

from e4_03_term_weighting import WEIGHTINGS


def compare_five_method_retrieval(index, query: str, limit: int = 10) -> dict:
    """Return original top-K hits, immutable PMID markers, and rank-overlap summary.

    The query, corpus, preprocessing, Cosine metric and limit are identical
    for all methods. Positive-score hit policy and tie breaks are M06's.
    """
    if limit != 10:
        raise ValueError("Project 3 A1 requires Top 10")
    rankings = {
        method: index.search(query, method, "Cosine", limit=limit)
        for method in WEIGHTINGS
    }
    # Pooled IDs in first-seen order, using the single canonical hit records.
    pmid_to_index = {}
    for method in WEIGHTINGS:
        result = rankings[method]
        if result["method"] != method or result["query"] != query or result["metric"] != "Cosine":
            raise ValueError("Retrieval engine returned inconsistent experiment settings")
        for position, hit in enumerate(result["hits"], 1):
            if hit["rank"] != position:
                raise ValueError(f"Inconsistent rank from retrieval engine: {method}")
            pmid = str(hit["pmid"])
            doc_index = int(hit["doc_index"])
            if pmid in pmid_to_index and pmid_to_index[pmid] != doc_index:
                raise ValueError(f"PMID {pmid} maps to conflicting corpus records")
            pmid_to_index[pmid] = doc_index

    pmids = list(pmid_to_index)
    labels = {pmid: f"D{i:02d}" for i, pmid in enumerate(pmids, 1)}
    per_method_top3 = {
        method: [str(hit["pmid"]) for hit in rankings[method]["hits"][:3]]
        for method in WEIGHTINGS
    }
    per_method_top10 = {
        method: [str(hit["pmid"]) for hit in rankings[method]["hits"]]
        for method in WEIGHTINGS
    }
    top1 = {method: (ids[0] if ids else None) for method, ids in per_method_top10.items()}
    top3_occurrences = {pmid: sum(pmid in ids for ids in per_method_top3.values())
                        for pmid in set().union(*(set(v) for v in per_method_top3.values()))}
    all_top3 = set.intersection(*(set(v) for v in per_method_top3.values()))
    overlaps = [
        {
            "Method A": a,
            "Method B": b,
            "Top 3 shared": len(set(per_method_top3[a]) & set(per_method_top3[b])),
            "Top 10 shared": len(set(per_method_top10[a]) & set(per_method_top10[b])),
        }
        for a, b in combinations(WEIGHTINGS, 2)
    ]
    top1_valid = [pmid for pmid in top1.values() if pmid is not None]
    counts = Counter(top1_valid)
    top1_max = max(counts.values(), default=0)
    # May be multiple tied largest groups. Preserve method order for readability.
    agreement_pmids = [pmid for pmid in dict.fromkeys(top1_valid)
                       if counts[pmid] == top1_max] if top1_max else []
    top1_groups = [
        {"pmid": pmid, "methods": [method for method in WEIGHTINGS
                                   if top1[method] == pmid]}
        for pmid in agreement_pmids
    ]
    summary = {
        "top1_max_agreement": top1_max,
        "top1_unique_pmids": len(counts),
        "top1_agreement_groups": top1_groups,
        "all_five_top1_same": all(pmid is not None for pmid in top1.values())
            and len(set(top1.values())) == 1,
        "top1_by_method": top1,
        "top3_repeated_pmids": sorted(
            (pmid for pmid, times in top3_occurrences.items() if times >= 2),
            key=lambda pmid: int(labels[pmid][1:])),
        "top3_shared_by_all_five": sorted(all_top3, key=lambda pmid: int(labels[pmid][1:])),
        "unique_top3_pmids": len(top3_occurrences),
        "unique_top10_pmids": len(pmids),
        "pairwise_overlap": overlaps,
    }
    return {"rankings": rankings, "pmid_to_index": pmid_to_index,
            "labels": labels, "summary": summary}


def selected_pmid_after_query_change(selected_pmid: str | None, source: str,
                                     valid_pmids, context_changed: bool):
    """Never silently switch to a different document after a query change.

    Returns (selected PMID or None, selection_expired). A manually specified
    PMID deliberately remains valid across query changes.
    """
    if source == "PMID" or not context_changed or selected_pmid is None:
        return selected_pmid, False
    if str(selected_pmid) in set(valid_pmids):
        return str(selected_pmid), False
    return None, True
