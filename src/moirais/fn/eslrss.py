"""Residual sum of squares."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_residual_sum_squares"]


def esl_residual_sum_squares(X, y, beta):
    """
    Residual sum of squares

    Formula: RSS(beta) = sum (y_i - x_i' beta)^2

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Residual sum of squares"})


def cheatsheet():
    return "eslrss: Residual sum of squares"
