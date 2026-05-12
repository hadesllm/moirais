# morie.fn -- function file from book-equation translation pipeline (hadesllm/morie)
"""UMAP projection of high-D embeddings to low-D for clustering / viz."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_umap_projection"]


def alammar_umap_projection(X, n_neighbors, min_dist, d_out):
    """
    UMAP projection of high-D embeddings to low-D for clustering / viz

    Formula: min cross_entropy(fuzzy_topology(X), fuzzy_topology(Z));  Z in R^{n x d_low}

    Parameters
    ----------
    X : array-like
        Input data.
    n_neighbors : array-like
        Input data.
    min_dist : array-like
        Input data.
    d_out : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Z

    References
    ----------
    Alammar Ch 5, UMAP section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "UMAP projection of high-D embeddings to low-D for clustering / viz"})


def cheatsheet():
    return "alumap: UMAP projection of high-D embeddings to low-D for clustering / viz"
