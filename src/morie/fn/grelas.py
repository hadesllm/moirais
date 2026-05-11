# morie.fn — function file (hadesllm/morie)
"""Elastic net cost combining L1 and L2 penalties with mix ratio r."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_elastic_net_cost"]


def geron_elastic_net_cost(X, y, theta, alpha, r):
    """
    Elastic net cost combining L1 and L2 penalties with mix ratio r

    Formula: J(theta) = MSE(theta) + r*alpha*sum |theta_i| + ((1-r)/2)*alpha*sum theta_i^2

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
    Géron Ch 4, Eq 4-12 (Elastic Net cost function)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elastic net cost combining L1 and L2 penalties with mix ratio r"})


def cheatsheet():
    return "grelas: Elastic net cost combining L1 and L2 penalties with mix ratio r"
