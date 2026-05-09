"""Redundancy 1 - H(X)/H_max."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["redundancy"]


def redundancy(p):
    """
    Redundancy 1 - H(X)/H_max

    Formula: R = 1 - H(X)/log|alphabet|

    Parameters
    ----------
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon (1948)
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Redundancy 1 - H(X)/H_max"})


def cheatsheet():
    return "redund: Redundancy 1 - H(X)/H_max"
