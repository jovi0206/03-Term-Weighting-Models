"""M05: Dot/cosine mathematical scoring, deliberately independent of ranking.

Cosine and explicit L2-dot MUST be identical up to floating point rounding.
Raw dot is included to observe the document-length effect in RQ4.
"""
import numpy as np
from project3_config import METRICS


def score_documents(index, query_vector, method, metric="Cosine"):
    if metric not in METRICS:
        raise ValueError(metric)
    matrix, norms = index.matrices[method], index.norms[method]
    qnorm = float(np.sqrt(query_vector.multiply(query_vector).sum()))
    raw = (matrix @ query_vector.T).toarray().ravel()
    if metric == "Raw Dot Product":
        scores = raw
    elif metric == "L2 Dot Product":
        inv_norm = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
        norm_matrix = matrix.multiply(inv_norm[:, None]).tocsr()
        unit_q = query_vector.multiply(1.0 / qnorm) if qnorm else query_vector
        scores = (norm_matrix @ unit_q.T).toarray().ravel()
    else:
        denominators = norms * qnorm
        scores = np.divide(raw, denominators, out=np.zeros_like(raw), where=denominators > 0)
    return scores, qnorm
