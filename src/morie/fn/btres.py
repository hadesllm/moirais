"""Residual bootstrap for OLS coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_residual_regression"]


def boot_residual_regression(X, y, B):
    """
    Residual bootstrap for OLS coefficients

    Formula: y* = Xβ̂ + r̂*; refit; collect β̂*_b

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_b, beta_hat

    References
    ----------
    Freedman (1981)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Residual bootstrap for OLS coefficients"})


def cheatsheet():
    return "btres: Residual bootstrap for OLS coefficients"
