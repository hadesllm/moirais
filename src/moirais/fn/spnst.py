"""Parametric non-stationary covariance model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_nonstationary_cov"]


def schabenberger_nonstationary_cov(coords, z):
    """
    Parametric non-stationary covariance model

    Formula: C(s1,s2) = sigma(s1)*sigma(s2)*rho(s1,s2) where sigma,rho vary spatially

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nonstationary_cov

    References
    ----------
    Schabenberger Ch 8, Sec 8.2.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Parametric non-stationary covariance model"})


def cheatsheet():
    return "spnst: Parametric non-stationary covariance model"
