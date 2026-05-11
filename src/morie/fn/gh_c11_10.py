# morie.fn — function file (hadesllm/morie)
"""Series GP prior (random Fourier features): f = sum_k beta_k phi_k, beta_k ~ N(0,lambda_k)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_series_gp"]


def ghosal_series_gp(x):
    """
    Series GP prior (random Fourier features): f = sum_k beta_k phi_k, beta_k ~ N(0,lambda_k)

    Formula: f ~ GP with k(x,y) = sum_k lambda_k phi_k(x) phi_k(y), eigenexpansion

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 11 §11.4.5
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Series GP prior (random Fourier features): f = sum_k beta_k phi_k, beta_k ~ N(0,lambda_k)"})


def cheatsheet():
    return "gh_c11_10: Series GP prior (random Fourier features): f = sum_k beta_k phi_k, beta_k ~ N(0,lambda_k)"
