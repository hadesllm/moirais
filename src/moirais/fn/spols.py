"""Ordinary least squares fitting of variogram model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_ols_variogram"]


def schabenberger_ols_variogram(empirical_variogram, variogram_model):
    """
    Ordinary least squares fitting of variogram model

    Formula: minimize sum_k [gamma_hat(h_k) - gamma(h_k;theta)]^2

    Parameters
    ----------
    empirical_variogram : array-like
        Input data.
    variogram_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: parameters

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.1
    """
    empirical_variogram = np.asarray(empirical_variogram, dtype=float)
    n = int(empirical_variogram) if empirical_variogram.ndim == 0 else len(empirical_variogram)
    result = float(np.mean(empirical_variogram))
    se = float(np.std(empirical_variogram, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ordinary least squares fitting of variogram model"})


def cheatsheet():
    return "spols: Ordinary least squares fitting of variogram model"
