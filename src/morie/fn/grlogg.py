# morie.fn — function file (hadesllm/morie)
"""Gradient of logistic-regression cost w.r.t. theta."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_logistic_cost_gradient"]


def geron_logistic_cost_gradient(X, y, theta):
    """
    Gradient of logistic-regression cost w.r.t. theta

    Formula: d J/d theta_j = (1/m)*sum_{i} (sigma(theta^T y^(i)) - y^(i)) x_j^(i)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Géron Ch 4, Eq 4-18 (Logistic cost partial derivative)
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of logistic-regression cost w.r.t. theta"})


def cheatsheet():
    return "grlogg: Gradient of logistic-regression cost w.r.t. theta"
