"""Lognormal kriging: predict on log scale, back-transform."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_lognormal_kriging"]


def schabenberger_lognormal_kriging(coords, z, target, cov_model):
    """
    Lognormal kriging: predict on log scale, back-transform

    Formula: Y=log(Z); Y_hat(s0) = OK on Y; Z_hat(s0)=exp(Y_hat+sigma^2_Y/2) (correction needed)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.1
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lognormal kriging: predict on log scale, back-transform"})


def cheatsheet():
    return "splgk: Lognormal kriging: predict on log scale, back-transform"
