"""Dropout regularization expectation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_dropout"]


def esl_dropout(X, p):
    """
    Dropout regularization expectation

    Formula: y = f(x .* m / p), m ~ Bernoulli(p)

    Parameters
    ----------
    X : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: regularizer

    References
    ----------
    Hastie ESL Ch 11
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dropout regularization expectation"})


def cheatsheet():
    return "esldat: Dropout regularization expectation"
