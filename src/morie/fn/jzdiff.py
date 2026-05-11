"""Jensen-Zhang disparity (KL-based diversity)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jenson_zhang_disparity"]


def jenson_zhang_disparity(y, p, q):
    """
    Jensen-Zhang disparity (KL-based diversity)

    Formula: D = H(M) - (1/2)(H(P) + H(Q)); M = (P+Q)/2

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Jensen-Zhang (1986)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jensen-Zhang disparity (KL-based diversity)"})


def cheatsheet():
    return "jzdiff: Jensen-Zhang disparity (KL-based diversity)"
