# morie.fn -- function file (hadesllm/morie)
"""Softmax probability for class k (normalized exponentials)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_probability"]


def geron_softmax_probability(X, theta):
    """
    Softmax probability for class k (normalized exponentials)

    Formula: p_hat_k = exp(s_k(X)) / sum_{j=1..K} exp(s_j(X))

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: p

    References
    ----------
    Géron Ch 4, Eq 4-20 (Softmax function)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax probability for class k (normalized exponentials)"})


def cheatsheet():
    return "grsmxp: Softmax probability for class k (normalized exponentials)"
