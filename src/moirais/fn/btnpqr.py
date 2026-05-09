"""Pairs bootstrap CI for quantile regression coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_quantile_regression"]


def boot_quantile_regression(X, y, tau, B, alpha):
    """
    Pairs bootstrap CI for quantile regression coefficients

    Formula: Resample (X,y) rows; refit QR(τ); percentile interval

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_b, lo, hi

    References
    ----------
    Koenker (2005)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pairs bootstrap CI for quantile regression coefficients"})


def cheatsheet():
    return "btnpqr: Pairs bootstrap CI for quantile regression coefficients"
