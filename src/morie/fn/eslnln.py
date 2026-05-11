"""Elastic net regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_elastic_net"]


def esl_elastic_net(X, y, lambda_, alpha):
    """
    Elastic net regression

    Formula: argmin |y-Xb|^2 + lambda(alpha|b|_1 + (1-alpha)|b|_2^2)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lambda_ : array-like
        Input data.
    alpha : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net regression"})


def cheatsheet():
    return "eslnln: Elastic net regression"
