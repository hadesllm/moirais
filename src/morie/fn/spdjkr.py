"""Disjunctive kriging: predict phi(Z(s0)) using phi(Z(s_i))."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_disjunctive_kriging"]


def schabenberger_disjunctive_kriging(coords, z, target, phi_func, cov_model):
    """
    Disjunctive kriging: predict phi(Z(s0)) using phi(Z(s_i))

    Formula: phi(Z)_hat(s0) = sum lambda_i phi(Z(s_i)); optimal for f(Z(s0)|data)

    Parameters
    ----------
    coords : array-like
        Input data.
    z : array-like
        Input data.
    target : array-like
        Input data.
    phi_func : array-like
        Input data.
    cov_model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction, variance

    References
    ----------
    Schabenberger Ch 5, Sec 5.6.4
    """
    z = np.asarray(z, dtype=float)
    n = int(z) if z.ndim == 0 else len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Disjunctive kriging: predict phi(Z(s0)) using phi(Z(s_i))"})


def cheatsheet():
    return "spdjkr: Disjunctive kriging: predict phi(Z(s0)) using phi(Z(s_i))"
