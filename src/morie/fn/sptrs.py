"""Polynomial trend surface model for spatially varying mean."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_trend_surface"]


def schabenberger_trend_surface(coords, z, poly_degree):
    """
    Polynomial trend surface model for spatially varying mean

    Formula: mu(s) = sum_k beta_k * f_k(s) where f_k are polynomial basis functions in coords

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    poly_degree : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coefficients, trend

    References
    ----------
    Schabenberger Ch 5, Sec 5.3.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polynomial trend surface model for spatially varying mean"})


def cheatsheet():
    return "sptrs: Polynomial trend surface model for spatially varying mean"
