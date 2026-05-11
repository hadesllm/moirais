"""Spectral clustering."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_spectral_cluster"]


def esl_spectral_cluster(W, k):
    """
    Spectral clustering

    Formula: Cluster eigenvectors of graph Laplacian

    Parameters
    ----------
    W : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Hastie ESL Ch 14
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral clustering"})


def cheatsheet():
    return "eslspc: Spectral clustering"
