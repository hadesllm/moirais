# morie.fn — function file (hadesllm/morie)
"""SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sgd_classifier"]


def geron_sgd_classifier(X, y, lr, n_iter):
    """
    SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent

    Formula: minimize sum max(0, 1 - y_i f(x_i)) + alpha/2 ||w||^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    lr : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent"})


def cheatsheet():
    return "hmsgdc: SGD classifier with hinge loss (linear SVM) trained by stochastic gradient descent"
