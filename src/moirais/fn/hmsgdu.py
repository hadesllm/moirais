# moirais.fn — function file (hadesllm/moirais)
"""SGD update using single random sample."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_sgd_update"]


def geron_sgd_update(X, y, theta, eta, seed):
    """
    SGD update using single random sample

    Formula: theta <- theta - eta_t * 2 x^(i) (x^(i)^T theta - y^(i))

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
    seed : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "SGD update using single random sample"})


def cheatsheet():
    return "hmsgdu: SGD update using single random sample"
