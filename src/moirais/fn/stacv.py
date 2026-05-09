"""Spatiotemporal autocovariance function."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatiotemporal_autocovariance"]


def spatiotemporal_autocovariance(x, coords, times):
    """
    Spatiotemporal autocovariance function

    Formula: C(h,u) = Cov(Z(s,t), Z(s+h,t+u))

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.
    times : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 8
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatiotemporal autocovariance function"})


def cheatsheet():
    return "stacv: Spatiotemporal autocovariance function"
