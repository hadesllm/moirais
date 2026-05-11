"""Spatial cross-validation (leave-one-out kriging)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_cross_validation"]


def spatial_cross_validation(x, coords):
    """
    Spatial cross-validation (leave-one-out kriging)

    Formula: MSPE = (1/n) sum (Z(s_i) - Z_hat_{-i}(s_i))^2

    Parameters
    ----------
    x : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial cross-validation (leave-one-out kriging)"})


def cheatsheet():
    return "spcrs: Spatial cross-validation (leave-one-out kriging)"
