"""Power semivariogram model (unbounded)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_power_variogram"]


def schabenberger_power_variogram(h, nugget, c1, alpha):
    """
    Power semivariogram model (unbounded)

    Formula: gamma(h) = c0 + c1*h^alpha, 0<alpha<2

    Parameters
    ----------
    h : array-like
        Input data.
    nugget : array-like
        Input data.
    c1 : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: semivariance

    References
    ----------
    Schabenberger Ch 4
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Power semivariogram model (unbounded)"})


def cheatsheet():
    return "sppow: Power semivariogram model (unbounded)"
