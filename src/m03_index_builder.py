"""M03: Build TF/CF/DF statistics and sparse document weight matrices.

This module does NOT parse XML, process queries or choose result presentation.
Method implementations live in E4; all algorithms use the same vocabulary.
"""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix

from project3_config import CORPUS_PATH, CONDITIONS
from project3_contracts import IndexData
from m01_corpus_reader import read_corpus, sha256
from m02_text_processor import tokens, idf
from e4_03_term_weighting import tf_weight, WEIGHTINGS


def build_index(corpus_path=None, condition="C"):
    """Return one contract for all downstream retrieval and evaluation modules."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    corpus_path = CORPUS_PATH if corpus_path is None else corpus_path
    docs, header = read_corpus(corpus_path)
    counters = [Counter(tokens(d["abstract"], condition)) for d in docs]
    cf, df = Counter(), Counter()
    for counts in counters:
        cf.update(counts)
        df.update(counts.keys())
    vocabulary = {term: col for col, term in enumerate(sorted(df))}
    n = len(docs)
    idfs = np.array([idf(n, df[term]) for term in vocabulary], dtype=np.float64)
    matrices, norms = {}, {}
    for method in WEIGHTINGS:
        rows, cols, values = [], [], []
        for row, counts in enumerate(counters):
            for term, frequency in counts.items():
                col = vocabulary[term]
                weight = tf_weight(frequency, method, idfs[col])
                if weight:
                    rows.append(row)
                    cols.append(col)
                    values.append(weight)
        matrix = csr_matrix((values, (rows, cols)), shape=(n, len(vocabulary)), dtype=np.float64)
        matrices[method] = matrix
        norms[method] = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    return IndexData(docs, condition, header, sha256(corpus_path), counters,
                     vocabulary, dict(df), dict(cf), idfs, matrices, norms)
