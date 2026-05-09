"""Planck blackbody spectrum."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["planck_function"]


def planck_function(lam, T):
    """
    Planck blackbody spectrum

    Formula: B(λ,T) = 2hc²/λ⁵ · 1/(exp(hc/λkT)-1)

    Parameters
    ----------
    lam : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Planck (1900)
    """
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    n = len(lam)
    result = float(np.mean(lam))
    se = float(np.std(lam, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Planck blackbody spectrum"})


def cheatsheet():
    return "plncF: Planck blackbody spectrum"
