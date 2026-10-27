"""Shared data contract (not an algorithm): read-only interface conventions.

Future Project 3 E-modules can consume IndexData fields without importing the
M08 controller or Streamlit UI. Most structures match Project 3 v1 exactly.
"""
from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class IndexData:
    docs: list
    condition: str
    header: dict
    corpus_sha256: str
    counters: list
    vocabulary: dict
    df: dict
    cf: dict
    idfs: np.ndarray
    matrices: dict[str, csr_matrix]
    norms: dict[str, np.ndarray]
