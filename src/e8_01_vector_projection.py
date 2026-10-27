"""E8-01: Two-dimensional visual exploration of sparse document vectors.

This is an OPTIONAL projection/exploration module, not a retrieval method.
Actual relevance ranks are always computed in M05/M06 in the original
high-dimensional term space.  The sparse matrix is L2-normalized before
TruncatedSVD so that cosine-oriented documents provide the input geometry.
A query is mapped with the *same trained transform*, unlike t-SNE which has
no standard out-of-sample transform.

This extension depends only on the IndexData contract and M04 query-vector
logic; it does not import the M08 controller or the Streamlit UI.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags
from sklearn.decomposition import TruncatedSVD

from e4_03_term_weighting import WEIGHTINGS
from m04_query_processor import query_vector


@dataclass(frozen=True)
class Projection2D:
    """Shared 2-D projection basis for one preprocessing/weighting selection."""
    method: str
    condition: str
    corpus_sha256: str
    coordinates: np.ndarray  # (documents, 2) - read-only by convention
    model: TruncatedSVD
    explained_variance_ratio: np.ndarray


def fit_document_projection(index, method: str, random_state: int = 42) -> Projection2D:
    """Fit deterministic 2-D TruncatedSVD to L2-normalized *sparse* documents.

    This model is for explanation only; do NOT replace high-dimensional
    cosine scores with distances on this 2-D plot. It is designed for
    documents with sufficiently many distinct terms (>=2 dimensions).
    """
    if method not in WEIGHTINGS:
        raise ValueError(f"Unknown weighting method: {method}")
    original = index.matrices[method]
    if original.shape[0] < 2 or original.shape[1] < 2:
        raise ValueError("2-D projection requires at least two documents and terms")
    norms = np.asarray(index.norms[method], dtype=np.float64)
    reciprocal = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    normalized = diags(reciprocal) @ original
    model = TruncatedSVD(n_components=2, n_iter=7, random_state=random_state)
    coordinates = np.asarray(model.fit_transform(normalized), dtype=np.float64)
    if coordinates.shape != (original.shape[0], 2):
        raise RuntimeError("Unexpected 2-D projection shape")
    return Projection2D(method, index.condition, index.corpus_sha256, coordinates,
                        model, np.asarray(model.explained_variance_ratio_, dtype=float))


def project_query(index, projection: Projection2D, query: str):
    """Apply the fitted document projection to an equally weighted query.

    Returns (xy coordinates or None, out-of-vocabulary terms). Zero vectors
    have no defined cosine direction, so they are not placed at the origin.
    """
    if (projection.condition != index.condition
            or projection.corpus_sha256 != index.corpus_sha256):
        raise ValueError("Projection was trained for a different corpus/condition")
    vector, oov, _ = query_vector(index, query, projection.method)
    qnorm = float(np.sqrt(vector.multiply(vector).sum()))
    if qnorm == 0:
        return None, oov
    unit_vector = vector.multiply(1.0 / qnorm).tocsr()
    xy = np.asarray(projection.model.transform(unit_vector)[0], dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise RuntimeError("Invalid query projection")
    return xy, oov
