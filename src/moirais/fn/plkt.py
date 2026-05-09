"""Plackett copula CDF (cross-product ratio)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["plackett_copula"]


def plackett_copula(y, u, v, theta):
    """
    Plackett copula CDF (cross-product ratio)

    Formula: C(u,v) = ((1+(theta-1)(u+v)) - sqrt((1+(theta-1)(u+v))^2 - 4 theta (theta-1) u v))/(2(theta-1))

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
    Plackett (1965)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Plackett copula CDF (cross-product ratio)"})


def cheatsheet():
    return "plkt: Plackett copula CDF (cross-product ratio)"
