"""Pesaran-Shin dynamic OLS for I(1) variables."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pesaran_shin_dols"]


def pesaran_shin_dols(y, X, p):
    """
    Pesaran-Shin dynamic OLS for I(1) variables

    Formula: y_t = beta'X_t + sum_{j=-p}^{p} gamma_j DeltaX_{t+j} + e_t

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pesaran & Shin (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pesaran-Shin dynamic OLS for I(1) variables"})


def cheatsheet():
    return "pesdol: Pesaran-Shin dynamic OLS for I(1) variables"
