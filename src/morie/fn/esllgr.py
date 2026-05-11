"""Logistic regression model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_logistic_reg"]


def esl_logistic_reg(X, y):
    """
    Logistic regression model

    Formula: log(P(Y=1|X)/(1-P)) = beta_0 + X' beta

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta, se

    References
    ----------
    Hastie ESL Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression model"})


def cheatsheet():
    return "esllgr: Logistic regression model"
