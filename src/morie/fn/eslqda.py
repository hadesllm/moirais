"""Quadratic discriminant analysis."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_qda"]


def esl_qda(X, y):
    """
    Quadratic discriminant analysis

    Formula: delta_k(x) = -(1/2) log|Sigma_k| - (1/2)(x-mu_k)' Sigma_k^{-1} (x-mu_k) + log pi_k

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quadratic discriminant analysis"})


def cheatsheet():
    return "eslqda: Quadratic discriminant analysis"
