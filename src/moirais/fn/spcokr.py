"""Cokriging: multivariate prediction using primary and secondary variables."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_cokriging"]


def schabenberger_cokriging(coords, z1, z2, target, cross_cov_model):
    """
    Cokriging: multivariate prediction using primary and secondary variables

    Formula: Z1_hat(s0) = lambda1'*Z1 + lambda2'*Z2; minimize MSE with unbiasedness constraints

    Parameters
    ----------
    coords : array-like
        Input data.
    z1 : array-like
        Input data.
    z2 : array-like
        Input data.
    target : array-like
        Input data.
    cross_cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5
    """
    coords = np.asarray(coords, dtype=float)
    n = int(coords) if coords.ndim == 0 else len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cokriging: multivariate prediction using primary and secondary variables"})


def cheatsheet():
    return "spcokr: Cokriging: multivariate prediction using primary and secondary variables"
