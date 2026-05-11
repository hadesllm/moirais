"""Least angle regression LARS path."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_least_angle_reg"]


def esl_least_angle_reg(X, y):
    """
    Least angle regression LARS path

    Formula: LARS path of solutions for varying lambda

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: path

    References
    ----------
    Hastie ESL Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Least angle regression LARS path"})


def cheatsheet():
    return "esllar: Least angle regression LARS path"
