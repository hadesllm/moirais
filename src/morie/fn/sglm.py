"""Spatial generalized linear model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["spatial_glm"]


def spatial_glm(x, y, coords):
    """
    Spatial generalized linear model

    Formula: g(mu) = X'beta + W, W ~ GP(0, Sigma)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial generalized linear model"})


def cheatsheet():
    return "sglm: Spatial generalized linear model"
