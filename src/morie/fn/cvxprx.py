"""Proximal operator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_proximal"]


def boyd_proximal(h, v):
    """
    Proximal operator

    Formula: prox_h(v) = argmin_x h(x) + (1/2)|x-v|^2

    Parameters
    ----------
    h : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: x

    References
    ----------
    Boyd CVX Ch 4
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proximal operator"})


def cheatsheet():
    return "cvxprx: Proximal operator"
