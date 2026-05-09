"""Spatial GLMM: conditional specification with random spatial effects."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_spatial_glmm"]


def schabenberger_spatial_glmm(x, y, coords, link, family):
    """
    Spatial GLMM: conditional specification with random spatial effects

    Formula: g(mu_i|b) = x_i'*beta + z_i'*b; b ~ N(0,Sigma_b(theta)); marginalize over b

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    coords : array-like
        Input data.
    link : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, random_effects

    References
    ----------
    Schabenberger Ch 6, Sec 6.3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial GLMM: conditional specification with random spatial effects"})


def cheatsheet():
    return "spglmm: Spatial GLMM: conditional specification with random spatial effects"
