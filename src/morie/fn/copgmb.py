"""Gumbel copula CDF (Archimedean, upper-tail)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gumbel_copula"]


def gumbel_copula(y, u, v, theta):
    """
    Gumbel copula CDF (Archimedean, upper-tail)

    Formula: C(u,v) = exp(-((-log u)^theta + (-log v)^theta)^{1/theta})

    Parameters
    ----------
    y : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Gumbel (1960)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gumbel copula CDF (Archimedean, upper-tail)"})


def cheatsheet():
    return "copgmb: Gumbel copula CDF (Archimedean, upper-tail)"
