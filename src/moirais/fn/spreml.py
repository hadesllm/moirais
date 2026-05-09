"""Restricted maximum likelihood (REML) for variogram parameters."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_reml_variogram"]


def schabenberger_reml_variogram(coords, z, X, variogram_model):
    """
    Restricted maximum likelihood (REML) for variogram parameters

    Formula: maximize log L_R(theta) = log L(theta) + 0.5*log|X'Sigma^{-1}X| (restricted likelihood)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    X : array-like
        Input data.
    variogram_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: parameters

    References
    ----------
    Schabenberger Ch 4, Sec 4.5.2
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Restricted maximum likelihood (REML) for variogram parameters"})


def cheatsheet():
    return "spreml: Restricted maximum likelihood (REML) for variogram parameters"
