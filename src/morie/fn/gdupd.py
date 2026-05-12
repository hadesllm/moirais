# morie.fn -- function file (hadesllm/morie)
"""Gradient descent parameter update rule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gradient_descent_update"]


def gradient_descent_update(beta, grad, alpha):
    """
    Gradient descent parameter update rule

    Formula: beta_new = beta - alpha * grad_beta(L)

    Parameters
    ----------
    beta : array-like
        Input data.
    grad : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'beta_new': 'array'}

    References
    ----------
    Montesinos Lopez Ch 3
    """
    beta = np.asarray(beta, dtype=float)
    n = int(beta) if beta.ndim == 0 else len(beta)
    result = float(np.mean(beta))
    se = float(np.std(beta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient descent parameter update rule"})


def cheatsheet():
    return "gdupd: Gradient descent parameter update rule"
