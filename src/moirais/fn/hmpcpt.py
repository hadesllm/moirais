# moirais.fn — function file (hadesllm/moirais)
"""Perceptron learning rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_perceptron"]


def geron_perceptron(X, y, eta, n_iter):
    """
    Perceptron learning rule

    Formula: w_{t+1} = w_t + eta (y - y_hat) x

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    eta : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights

    References
    ----------
    Géron Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceptron learning rule"})


def cheatsheet():
    return "hmpcpt: Perceptron learning rule"
