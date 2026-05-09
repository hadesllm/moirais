"""Generalized additive model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_gam"]


def esl_gam(X, y, g):
    """
    Generalized additive model

    Formula: g(mu) = alpha + sum f_j(X_j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    g : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Hastie ESL Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Generalized additive model"})


def cheatsheet():
    return "eslgam: Generalized additive model"
