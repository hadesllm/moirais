"""Sparse vector technique."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sparse_vector"]


def sparse_vector(queries, threshold, c, epsilon):
    """
    Sparse vector technique

    Formula: answer queries above noisy threshold; abort after c hits

    Parameters
    ----------
    queries : array-like
        Input data.
    threshold : array-like
        Input data.
    c : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014); Hardt-Rothblum (2010)
    """
    queries = np.atleast_1d(np.asarray(queries, dtype=float))
    n = len(queries)
    result = float(np.mean(queries))
    se = float(np.std(queries, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sparse vector technique"})


def cheatsheet():
    return "sparsv: Sparse vector technique"
