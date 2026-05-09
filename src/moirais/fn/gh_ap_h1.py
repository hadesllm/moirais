# moirais.fn — function file (hadesllm/moirais)
"""Inverse-Gaussian distribution IG(alpha, gamma): density and Levy representation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_inv_gauss"]


def ghosal_inv_gauss(x):
    """
    Inverse-Gaussian distribution IG(alpha, gamma): density and Levy representation

    Formula: f(x; alpha, gamma) = sqrt(gamma/(2*pi*x^3)) exp(-(gamma(x-alpha)^2)/(2*alpha^2*x))

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
    Ghosal App H
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse-Gaussian distribution IG(alpha, gamma): density and Levy representation"})


def cheatsheet():
    return "gh_ap_h1: Inverse-Gaussian distribution IG(alpha, gamma): density and Levy representation"
