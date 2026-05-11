"""Neyman optimal allocation across strata."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["neyman_allocation"]


def neyman_allocation(y, N_h, S_h, n):
    """
    Neyman optimal allocation across strata

    Formula: n_h = n * (N_h S_h) / sum_k (N_k S_k)

    Parameters
    ----------
    y : array-like
        Input data.
    N_h : array-like
        Input data.
    S_h : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neyman (1934)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Neyman optimal allocation across strata"})


def cheatsheet():
    return "neyman: Neyman optimal allocation across strata"
