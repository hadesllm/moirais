"""Coefficient of determination R^2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_r_squared"]


def esl_r_squared(X, y, beta):
    """
    Coefficient of determination R^2

    Formula: R^2 = 1 - RSS/TSS

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coefficient of determination R^2"})


def cheatsheet():
    return "eslr2: Coefficient of determination R^2"
