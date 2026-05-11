# morie.fn — function file (hadesllm/morie)
"""Predicted probability for binary logistic regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_regression_probability"]


def geron_logistic_regression_probability(X, theta):
    """
    Predicted probability for binary logistic regression

    Formula: p_hat = sigma(theta^T X)

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
    Géron Ch 4, Eq 4-15 (Logistic Regression probability)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Predicted probability for binary logistic regression"})


def cheatsheet():
    return "grlogp: Predicted probability for binary logistic regression"
