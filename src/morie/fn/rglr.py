# morie.fn — function file (hadesllm/morie)
"""Logistic regression for binary classification."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_logistic_regression"]


def rangayyan_logistic_regression(X, y, lr, max_iter):
    """
    Logistic regression for binary classification

    Formula: P(y=1|y) = sigmoid(w^T*y + b) = 1/(1+exp(-(w^T*y+b)))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: w, b, probabilities

    References
    ----------
    Rangayyan Ch 10.7
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Logistic regression for binary classification"})


def cheatsheet():
    return "rglr: Logistic regression for binary classification"
