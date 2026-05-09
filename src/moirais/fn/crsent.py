"""Cross entropy H(p, q)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cross_entropy"]


def cross_entropy(p, q, base):
    """
    Cross entropy H(p, q)

    Formula: -sum p log q

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.
    base : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross entropy H(p, q)"})


def cheatsheet():
    return "crsent: Cross entropy H(p, q)"
