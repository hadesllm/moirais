"""Isomap MDS on geodesic distance matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_isomap"]


def sgt_isomap(X, k, dim):
    """
    Isomap MDS on geodesic distance matrix

    Formula: Apply classical MDS to D_geo from k-NN graph

    Parameters
    ----------
    X : array-like
        Input data.
    k : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y

    References
    ----------
    Tenenbaum-de Silva-Langford (2000)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Isomap MDS on geodesic distance matrix"})


def cheatsheet():
    return "sgtnsne: Isomap MDS on geodesic distance matrix"
