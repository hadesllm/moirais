# moirais.fn — function file (hadesllm/moirais)
"""Batch gradient descent step on linear-regression MSE."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_gradient_descent"]


def geron_batch_gradient_descent(X, y, theta, eta, n_iter):
    """
    Batch gradient descent step on linear-regression MSE

    Formula: theta_{t+1} = theta_t - eta * (2/m) * X^T (X theta_t - y)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.
    eta : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4, Eq 4-6 (Batch Gradient Descent step)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch gradient descent step on linear-regression MSE"})


def cheatsheet():
    return "grbgd: Batch gradient descent step on linear-regression MSE"
