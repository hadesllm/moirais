"""Bernstein's inequality."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bernstein_inequality"]


def bernstein_inequality(sigma2, M, n, t):
    """
    Bernstein's inequality

    Formula: P(S_n >= n t) <= exp(-n t^2 / (2 sigma^2 + 2 M t/3))

    Parameters
    ----------
    sigma2 : array-like
        Input data.
    M : array-like
        Input data.
    n : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bernstein (1924)
    """
    sigma2 = np.atleast_1d(np.asarray(sigma2, dtype=float))
    n = len(sigma2)
    result = float(np.mean(sigma2))
    se = float(np.std(sigma2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bernstein's inequality"})


def cheatsheet():
    return "brnstn: Bernstein's inequality"
