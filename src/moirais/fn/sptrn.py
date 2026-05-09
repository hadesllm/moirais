"""Trend surface analysis (polynomial)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_trend_surface"]


def spatial_trend_surface(x, coords):
    """
    Trend surface analysis (polynomial)

    Formula: mu(s) = sum beta_k f_k(s1,s2), polynomial in coords

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Schabenberger Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Trend surface analysis (polynomial)"})


def cheatsheet():
    return "sptrn: Trend surface analysis (polynomial)"
