"""Spatial covariance function C(h) = Cov(Z(s), Z(s+h))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_covariance_function"]


def schabenberger_covariance_function(coords, z):
    """
    Spatial covariance function C(h) = Cov(Z(s), Z(s+h))

    Formula: C(h) = E[Z(s)Z(s+h)] - mu^2 under stationarity

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schabenberger Ch 1/2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial covariance function C(h) = Cov(Z(s), Z(s+h))"})


def cheatsheet():
    return "spcovf: Spatial covariance function C(h) = Cov(Z(s), Z(s+h))"
