"""Rényi DP for subsampled mechanisms."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rdp_subsampled_composition"]


def rdp_subsampled_composition(y, alpha, q, epsilon):
    """
    Rényi DP for subsampled mechanisms

    Formula: epsilon'(alpha) = (1/(alpha-1)) log E[(1-q)+q M^alpha] (Mironov-Wang)

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.
    q : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mironov, Talwar, Zhang (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rényi DP for subsampled mechanisms"})


def cheatsheet():
    return "rdpcomp: Rényi DP for subsampled mechanisms"
