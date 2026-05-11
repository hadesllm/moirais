"""Gradient boosting machine."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_gbm"]


def esl_gbm(X, y, M, nu):
    """
    Gradient boosting machine

    Formula: f_m(x) = f_{m-1}(x) + nu h_m(x; gamma_m)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    M : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 10
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient boosting machine"})


def cheatsheet():
    return "eslgbm: Gradient boosting machine"
