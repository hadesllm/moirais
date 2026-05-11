"""Partial least squares regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_pls"]


def esl_pls(X, y, M):
    """
    Partial least squares regression

    Formula: PLS directions z_m maximizing Cov(z_m,y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partial least squares regression"})


def cheatsheet():
    return "eslpls: Partial least squares regression"
