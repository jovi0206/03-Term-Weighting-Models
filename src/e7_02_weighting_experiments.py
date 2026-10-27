"""E7: Controlled ranking and normalization comparison for RQ1–RQ4.

A separate extension so future assignment experiments need not alter M06.
"""


def compare_rankings(index, query, method1="Raw TF", method2="Raw TF-IDF", k=10):
    first = index.search(query, method1, limit=k)["hits"]
    second = index.search(query, method2, limit=k)["hits"]
    ranks_a = {hit["pmid"]: hit["rank"] for hit in first}
    ranks_b = {hit["pmid"]: hit["rank"] for hit in second}
    pmids = list(dict.fromkeys([hit["pmid"] for hit in first + second]))
    docs = {doc["pmid"]: doc for doc in index.docs}
    return [{"PMID": pmid, "Title": docs[pmid]["title"],
             method1: ranks_a.get(pmid), method2: ranks_b.get(pmid)} for pmid in pmids]


def compare_normalization(index, query, method="Raw TF-IDF", k=10):
    hits = {metric: index.search(query, method, metric, k)["hits"]
            for metric in ("Raw Dot Product", "Cosine", "L2 Dot Product")}
    return {
        "hits": hits,
        "raw_cosine_overlap": len({h["pmid"] for h in hits["Raw Dot Product"]}
                                  & {h["pmid"] for h in hits["Cosine"]}),
        "cosine_l2_same_order": [h["pmid"] for h in hits["Cosine"]]
                                == [h["pmid"] for h in hits["L2 Dot Product"]],
        "max_absolute_error": max((abs(a["score"] - b["score"])
                                   for a, b in zip(hits["Cosine"], hits["L2 Dot Product"])),
                                  default=0.0),
    }
