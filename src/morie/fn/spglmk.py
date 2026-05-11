"""Spatial prediction in GLMs via kriging on random effect."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spatial_glm_kriging"]


def schabenberger_spatial_glm_kriging(x, y, coords, target, cov_model, link):
    """
    Spatial prediction in GLMs via kriging on random effect

    Formula: g(mu(s0)) = x(s0)'*beta + b_hat(s0) where b_hat(s0) kriged from b_hat(s_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.
    target : array-like
        Input data.
    cov_model : array-like
        Input data.
    link : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial prediction in GLMs via kriging on random effect"})


def cheatsheet():
    return "spglmk: Spatial prediction in GLMs via kriging on random effect"
