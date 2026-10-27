"""E8: One fixed cosine-UMAP document map for comparing five actual VSM Top 10 lists.

The reference embedding is **always** Log TF-IDF under the *current*
preprocessing condition. Documents have a single 2-D coordinate regardless of
which weighting retrieved them. All rankings/scores come from M08, never UMAP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import diags

from e4_03_term_weighting import WEIGHTINGS

REFERENCE_WEIGHTING = "Log TF-IDF"
UMAP_NEIGHBORS = 25
UMAP_MIN_DIST = 0.15
UMAP_SEED = 42
UMAP_METRIC = "cosine"


@dataclass
class SharedUMAP:
    """One cached fit for one formal corpus/preprocessing combination."""
    coords: np.ndarray
    model: object
    valid_indices: np.ndarray
    parameters: dict


def _normalized_reference_docs(index):
    """Keep zero-vector documents out of cosine UMAP; never invent coordinates."""
    doc_weights = index.matrices[REFERENCE_WEIGHTING]
    norms = np.asarray(index.norms[REFERENCE_WEIGHTING], dtype=float)
    if doc_weights.shape[0] != len(index.docs) or len(norms) != len(index.docs):
        raise ValueError("Reference document matrix/document count mismatch")
    valid = np.flatnonzero(np.isfinite(norms) & (norms > 0))
    if len(valid) < 3:
        raise ValueError("Need at least three nonzero Log TF-IDF document vectors for cosine UMAP")
    # Preserve CSR sparsity (3K x 28K is too large to densify without reason).
    normalizer = diags(1.0 / norms[valid], format="csr")
    normalized = (normalizer @ doc_weights[valid]).tocsr().astype(np.float32)
    return valid, normalized


def fit_shared_umap(index):
    """Fit ONCE on reference document vectors; unrelated to scoring/ranking.

    Import lazily, so the other six UI pages can still run before umap-learn is
    installed. Streamlit controls caching in M09 by corpus SHA and condition.
    """
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError("UMAP requires 'umap-learn'. Run: python -m pip install -r requirements.txt") from exc

    valid, normalized = _normalized_reference_docs(index)
    kwargs = dict(n_components=2, n_neighbors=min(UMAP_NEIGHBORS, len(valid) - 1),
                  min_dist=UMAP_MIN_DIST, metric=UMAP_METRIC,
                  random_state=UMAP_SEED, transform_seed=UMAP_SEED,
                  low_memory=True, n_jobs=1)
    model = umap.UMAP(**kwargs)
    embedded = np.asarray(model.fit_transform(normalized), dtype=float)
    if embedded.shape != (len(valid), 2) or not np.isfinite(embedded).all():
        raise ValueError("UMAP returned invalid two-dimensional reference coordinates")
    coords = np.full((len(index.docs), 2), np.nan, dtype=float)
    coords[valid] = embedded
    return SharedUMAP(coords, model, valid, dict(
        corpus_documents=len(index.docs), plotted_documents=len(valid),
        reference_weighting=REFERENCE_WEIGHTING, metric=UMAP_METRIC,
        n_neighbors=kwargs["n_neighbors"], min_dist=UMAP_MIN_DIST,
        random_state=UMAP_SEED, n_components=2, normalization="L2",
        corpus_sha256=index.corpus_sha256, preprocessing=index.condition,
    ))


def project_shared_query(index, projection: SharedUMAP, query: str):
    """Transform a query in the SAME reference weighting and UMAP fit.

    A zero/OOV query cannot be projected. Return a reason, not (0,0).
    """
    if projection.parameters["corpus_sha256"] != index.corpus_sha256 or (
            projection.parameters["preprocessing"] != index.condition):
        raise ValueError("Query/index source differs from the cached UMAP fit")
    vector, oov, _ = index.query_vector(query, REFERENCE_WEIGHTING)
    norm = np.sqrt(float(vector.multiply(vector).sum()))
    if norm == 0:
        return None, ("No nonzero Log TF-IDF query vector in the formal corpus; "
                      "the query star is omitted (e.g. OOV or IDF=0).")
    normalized = (vector / norm).tocsr().astype(np.float32)
    transformed = np.asarray(projection.model.transform(normalized), dtype=float)
    if transformed.shape != (1, 2) or not np.isfinite(transformed).all():
        raise ValueError("UMAP could not project query into the reference space")
    return (float(transformed[0, 0]), float(transformed[0, 1])), None


def plot_inputs(index, projection: SharedUMAP, method: str, result: dict,
                labels: dict, query_xy=None):
    """Pure shape/identity prep; takes actual M08 hits without reranking."""
    if method not in WEIGHTINGS or result["method"] != method or result["metric"] != "Cosine":
        raise ValueError("UMAP overlays require unmodified, original Cosine search hits")
    if projection.coords.shape != (len(index.docs), 2):
        raise ValueError("Reference coordinates do not correspond to the indexed corpus")
    docids = []
    for hit in result["hits"]:
        position = int(hit["doc_index"])
        if position < 0 or position >= len(index.docs) or str(index.docs[position]["pmid"]) != str(hit["pmid"]):
            raise ValueError("Search hit and reference document do not match")
        docids.append(position)
    if len(docids) != len(set(docids)):
        raise ValueError("Top 10 contains a repeated document")
    top = set(docids)
    background = [i for i in projection.valid_indices if int(i) not in top]
    top_valid = []
    for hit in result["hits"]:
        position = int(hit["doc_index"])
        if np.isfinite(projection.coords[position]).all():
            top_valid.append({"doc_index": position, "pmid": str(hit["pmid"]),
                              "rank": int(hit["rank"]), "score": float(hit["score"]),
                              "title": str(hit["title"]),
                              "marker": labels.get(str(hit["pmid"]), "")})
    return {"other_indices": background, "top": top_valid, "query_xy": query_xy,
            "unmapped_top10": len(result["hits"]) - len(top_valid)}
