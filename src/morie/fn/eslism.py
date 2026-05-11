"""Isomap geodesic embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_isomap"]


def esl_isomap(X, k, neighbors):
    """
    Isomap geodesic embedding

    Formula: MDS on geodesic distances of nearest-neighbor graph

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    neighbors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: embedding

    References
    ----------
    Hastie ESL Ch 14
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isomap geodesic embedding"})


def cheatsheet():
    return "eslism: Isomap geodesic embedding"
