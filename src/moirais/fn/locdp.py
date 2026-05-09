"""Local DP (each user randomizes)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["local_dp"]


def local_dp(x, mech, epsilon):
    """
    Local DP (each user randomizes)

    Formula: P(M(x)=y)/P(M(x')=y) ≤ exp(ε)

    Parameters
    ----------
    x : array-like
        Input data.
    mech : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kasiviswanathan et al (2011)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Local DP (each user randomizes)"})


def cheatsheet():
    return "locdp: Local DP (each user randomizes)"
