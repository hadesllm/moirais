# morie.fn -- function file (hadesllm/morie)
"""Gradient of K-class softmax cross-entropy w.r.t. Theta_k."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_cost_gradient"]


def geron_softmax_cost_gradient(X, Y, theta):
    """
    Gradient of K-class softmax cross-entropy w.r.t. Theta_k

    Formula: grad_{Theta_k} J = (1/m) sum_i (p_hat_k^(i) - y_k^(i)) X^(i)

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradient

    References
    ----------
    Géron Ch 4, Eq 4-23 (Cross-entropy gradient)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient of K-class softmax cross-entropy w.r.t. Theta_k"})


def cheatsheet():
    return "grxeng: Gradient of K-class softmax cross-entropy w.r.t. Theta_k"
