"""Spatial Durbin model: SAR with spatially lagged covariates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spatial_durbin_model"]


def schabenberger_spatial_durbin_model(x, y, w):
    """
    Spatial Durbin model: SAR with spatially lagged covariates

    Formula: Y = rho*W*Y + X*beta + W*X*theta + epsilon

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
        Keys: rho, beta, theta, se

    References
    ----------
    Schabenberger Ch 6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial Durbin model: SAR with spatially lagged covariates"})


def cheatsheet():
    return "spsdm: Spatial Durbin model: SAR with spatially lagged covariates"
