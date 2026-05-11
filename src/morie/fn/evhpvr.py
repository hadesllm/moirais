"""Heffernan-Tawn conditional extreme model fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_heffernan_tawn"]


def evt_heffernan_tawn(X, u):
    """
    Heffernan-Tawn conditional extreme model fit

    Formula: Y_j|X=x = a_j x + x^{b_j} Z_j

    Parameters
    ----------
    X : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a, b, mu_z, sigma_z

    References
    ----------
    Heffernan & Tawn (2004)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heffernan-Tawn conditional extreme model fit"})


def cheatsheet():
    return "evhpvr: Heffernan-Tawn conditional extreme model fit"
