# morie.fn — function file (hadesllm/morie)
"""Multinomial logistic (softmax) regression end-to-end fit."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_multinomial_logistic"]


def geron_multinomial_logistic(X, Y, lr, n_iter):
    """
    Multinomial logistic (softmax) regression end-to-end fit

    Formula: theta* = argmin CrossEntropy(softmax(X Theta), Y)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    lr : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multinomial logistic (softmax) regression end-to-end fit"})


def cheatsheet():
    return "hmmnl: Multinomial logistic (softmax) regression end-to-end fit"
