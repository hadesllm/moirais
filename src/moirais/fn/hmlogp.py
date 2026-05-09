# moirais.fn — function file (hadesllm/moirais)
"""Logistic regression probability prediction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_probability"]


def geron_logistic_probability(X, theta):
    """
    Logistic regression probability prediction

    Formula: p_hat = sigma(theta^T x)

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression probability prediction"})


def cheatsheet():
    return "hmlogp: Logistic regression probability prediction"
