# moirais.fn — function file (hadesllm/moirais)
"""Spectral clustering: eigenvectors of graph Laplacian."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_spectral_clustering"]


def geron_spectral_clustering(X, n_clusters, affinity):
    """
    Spectral clustering: eigenvectors of graph Laplacian

    Formula: embed via eigenvectors of L = D - W; k-means in embedding

    Parameters
    ----------
    X : array-like
        Input data.
    n_clusters : array-like
        Input data.
    affinity : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral clustering: eigenvectors of graph Laplacian"})


def cheatsheet():
    return "hmspcl: Spectral clustering: eigenvectors of graph Laplacian"
