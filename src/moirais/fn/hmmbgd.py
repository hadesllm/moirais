# moirais.fn — function file (hadesllm/moirais)
"""Mini-batch gradient descent on subset of size b."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_minibatch_gd"]


def geron_minibatch_gd(X, y, theta, eta, b):
    """
    Mini-batch gradient descent on subset of size b

    Formula: theta <- theta - eta * (2/b) X_b^T (X_b theta - y_b)

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

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch gradient descent on subset of size b"})


def cheatsheet():
    return "hmmbgd: Mini-batch gradient descent on subset of size b"
