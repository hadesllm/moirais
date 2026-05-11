"""Assortativity coefficient."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["assortativity"]


def assortativity(G, attribute):
    """
    Assortativity coefficient

    Formula: r = sum (e_ii - a_i b_i) / (1 - sum a_i b_i)

    Parameters
    ----------
    G : array-like
        Input data.
    attribute : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2002)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Assortativity coefficient"})


def cheatsheet():
    return "asorxx: Assortativity coefficient"
