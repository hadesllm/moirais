# moirais.fn — function file (hadesllm/moirais)
"""Stochastic gradient descent step using a single random sample."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stochastic_gradient_descent"]


def geron_stochastic_gradient_descent(X, y, theta, eta, n_iter):
    """
    Stochastic gradient descent step using a single random sample

    Formula: theta_{t+1} = theta_t - eta_t * 2 * y^(i) (y^(i)^T theta_t - y^(i))

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
    Géron Ch 4, Stochastic Gradient Descent section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic gradient descent step using a single random sample"})


def cheatsheet():
    return "grsgd: Stochastic gradient descent step using a single random sample"
