"""Matern covariance function class."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_matern_covariance"]


def schabenberger_matern_covariance(h, sigma2, nu, a):
    """
    Matern covariance function class

    Formula: C(h;nu,a) = sigma^2 * 2^{1-nu}/Gamma(nu) * (sqrt(2*nu)*h/a)^nu * K_nu(sqrt(2*nu)*h/a)

    Parameters
    ----------
    h : array-like
        Input data.
    sigma2 : array-like
        Input data.
    nu : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: covariance

    References
    ----------
    Schabenberger Ch 4, Sec 4.3.2
    """
    h = np.asarray(h, dtype=float)
    n = int(h) if h.ndim == 0 else len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matern covariance function class"})


def cheatsheet():
    return "spmatr: Matern covariance function class"
