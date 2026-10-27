"""M06: Rank scored abstracts with stable tie-breaking and positive-score policy.

M06 receives a common IndexData-like contract; it does not import the UI,
controller or evaluation implementation.
"""
import numpy as np
from m04_query_processor import query_vector
from m05_similarity import score_documents


def search(index, query, method="Raw TF-IDF", metric="Cosine", limit=10):
    if limit < 1:
        raise ValueError("limit must be positive")
    vector, oov, counts = query_vector(index, query, method)
    scores, qnorm = score_documents(index, vector, method, metric)
    order = np.lexsort((np.arange(len(scores)), -scores))
    hits = []
    for doc_position in order:
        if scores[doc_position] <= 0 or len(hits) >= limit:
            break
        hits.append({
            "rank": len(hits) + 1,
            "doc_index": int(doc_position),
            "score": float(scores[doc_position]),
            **index.docs[doc_position],
            "length": int(sum(index.counters[doc_position].values())),
            "l2_norm": float(index.norms[method][doc_position]),
        })
    return {
        "query": query, "method": method, "metric": metric,
        "condition": index.condition, "hits": hits, "oov": oov,
        "query_terms": dict(counts), "query_l2_norm": qnorm,
        "nonzero_documents": int(np.count_nonzero(scores > 0)),
        "query_has_weights": bool(qnorm > 0),
    }
