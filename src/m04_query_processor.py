"""M04: Transform a user query into the document index vocabulary.

Policy: use exactly the same selected term weighting as documents; report
OOV terms and tokens before/after filtering separately.
"""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

from e4_03_term_weighting import tf_weight, WEIGHTINGS
from m02_text_processor import tokens


def query_vector(index, query, method):
    if method not in WEIGHTINGS:
        raise ValueError(method)
    counts = Counter(tokens(query, index.condition))
    columns, weights, oov = [], [], []
    for term, count in counts.items():
        col = index.vocabulary.get(term)
        if col is None:
            oov.append(term)
            continue
        weight = tf_weight(count, method, index.idfs[col])
        if weight:
            columns.append(col)
            weights.append(weight)
    vector = csr_matrix((weights, ([0] * len(columns), columns)),
                        shape=(1, len(index.vocabulary)), dtype=np.float64)
    return vector, oov, counts
