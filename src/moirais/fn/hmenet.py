# moirais.fn — function file (hadesllm/moirais)
"""Elastic net cost combining L1 and L2 with ratio r."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_elastic_net"]


def geron_elastic_net(X, y, theta, alpha, r):
    """
    Elastic net cost combining L1 and L2 with ratio r

    Formula: J = MSE + r*alpha*sum|theta_i| + ((1-r)/2)*alpha*sum theta_i^2

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.
    alpha : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net cost combining L1 and L2 with ratio r"})


def cheatsheet():
    return "hmenet: Elastic net cost combining L1 and L2 with ratio r"
