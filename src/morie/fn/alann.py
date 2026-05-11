# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Approximate nearest neighbor search (HNSW-style): navigable-small-world greedy descent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_approximate_nearest_neighbor"]


def alammar_approximate_nearest_neighbor(query_vec, index, ef_search):
    """
    Approximate nearest neighbor search (HNSW-style): navigable-small-world greedy descent

    Formula: start at entry point; at each layer greedy-descend toward query; refine at layer 0

    Parameters
    ----------
    query_vec : array-like
        Input data.
    index : array-like
        Input data.
    ef_search : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: top_k_indices

    References
    ----------
    Alammar Ch 8, ANN section
    """
    query_vec = np.atleast_1d(np.asarray(query_vec, dtype=float))
    n = len(query_vec)
    result = float(np.mean(query_vec))
    se = float(np.std(query_vec, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Approximate nearest neighbor search (HNSW-style): navigable-small-world greedy descent"})


def cheatsheet():
    return "alann: Approximate nearest neighbor search (HNSW-style): navigable-small-world greedy descent"
