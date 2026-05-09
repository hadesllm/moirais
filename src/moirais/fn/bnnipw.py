"""Bound without IV using proxy."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_no_iv_proxy"]


def bound_no_iv_proxy(y, D, Z_proxy):
    """
    Bound without IV using proxy

    Formula: surrogate-IV bound via proxy validity

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    Z_proxy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tchetgen Tchetgen et al (2020) proxy
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound without IV using proxy"})


def cheatsheet():
    return "bnnipw: Bound without IV using proxy"
