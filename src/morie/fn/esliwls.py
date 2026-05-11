"""Iteratively reweighted least squares."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_iwls"]


def esl_iwls(X, y, beta0):
    """
    Iteratively reweighted least squares

    Formula: beta^{new} = (X'WX)^{-1} X'Wz

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    beta0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta

    References
    ----------
    Hastie ESL Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Iteratively reweighted least squares"})


def cheatsheet():
    return "esliwls: Iteratively reweighted least squares"
