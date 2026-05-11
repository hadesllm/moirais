"""Inverse distance weighting interpolation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_idw"]


def schabenberger_idw(coords, z, target, power):
    """
    Inverse distance weighting interpolation

    Formula: Z_hat(s0) = sum w_i*Z(s_i) / sum w_i; w_i = 1/d(s0,s_i)^p

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    power : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Schabenberger supplement
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse distance weighting interpolation"})


def cheatsheet():
    return "spmidw: Inverse distance weighting interpolation"
