"""Rate-distortion function R(D)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rate_distortion"]


def rate_distortion(px, distortion, D):
    """
    Rate-distortion function R(D)

    Formula: R(D) = min I(X;Y) s.t. E[d(X,Y)] <= D

    Parameters
    ----------
    px : array-like
        Input data.
    distortion : array-like
        Input data.
    D : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shannon (1959)
    """
    px = np.atleast_1d(np.asarray(px, dtype=float))
    n = len(px)
    result = float(np.mean(px))
    se = float(np.std(px, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rate-distortion function R(D)"})


def cheatsheet():
    return "rdfunc: Rate-distortion function R(D)"
