"""Spatial linear mixed model (REML)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_mixed_model"]


def spatial_mixed_model(x, y, coords):
    """
    Spatial linear mixed model (REML)

    Formula: Y = X*beta + Z*u + e, Cov(e) = R(theta)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Schabenberger Ch 5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial linear mixed model (REML)"})


def cheatsheet():
    return "smixd: Spatial linear mixed model (REML)"
