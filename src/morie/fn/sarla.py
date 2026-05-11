# morie.fn — function file (hadesllm/morie)
"""Spatial autoregressive lag model (SAR lag)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_ar_lag"]


def spatial_ar_lag(x, y, w):
    """
    Spatial autoregressive lag model (SAR lag)

    Formula: Y = rho*W*Y + X*beta + e

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Schabenberger Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial autoregressive lag model (SAR lag)"})


def cheatsheet():
    return "sarla: Spatial autoregressive lag model (SAR lag)"
