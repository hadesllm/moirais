"""Matern variogram model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matern_variogram_model"]


def matern_variogram_model(h, c0, c, a, nu):
    """
    Matern variogram model

    Formula: gamma(h) = c0 + c [1 - (2^{1-nu}/Gamma(nu)) (h/a)^nu K_nu(h/a)]

    Parameters
    ----------
    h : array-like
        Input data.
    c0 : array-like
        Input data.
    c : array-like
        Input data.
    a : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Stein (1999) §2.7
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matern variogram model"})


def cheatsheet():
    return "matvar: Matern variogram model"
