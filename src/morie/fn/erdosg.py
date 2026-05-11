"""Erdős-Rényi G(n,p)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["erdos_renyi_gnp"]


def erdos_renyi_gnp(n, p):
    """
    Erdős-Rényi G(n,p)

    Formula: each edge independent w.p. p

    Parameters
    ----------
    n : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Erdős-Rényi (1959)
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Erdős-Rényi G(n,p)"})


def cheatsheet():
    return "erdosg: Erdős-Rényi G(n,p)"
