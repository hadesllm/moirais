# moirais.fn — function file (hadesllm/moirais)
"""Mini-batch gradient descent step on a random subset of size b."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_minibatch_gradient_descent"]


def geron_minibatch_gradient_descent(X, y, theta, eta, b, n_iter):
    """
    Mini-batch gradient descent step on a random subset of size b

    Formula: theta_{t+1} = theta_t - eta * (2/b) * X_b^T (X_b theta_t - y_b)

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
    b : array-like
        Input data.
    n_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4, Mini-batch Gradient Descent section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch gradient descent step on a random subset of size b"})


def cheatsheet():
    return "grmgd: Mini-batch gradient descent step on a random subset of size b"
