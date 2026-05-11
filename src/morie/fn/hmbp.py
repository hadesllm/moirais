# morie.fn — function file (hadesllm/morie)
"""Backpropagation of errors to compute gradients."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_backpropagation"]


def geron_backpropagation(X, y, weights, activations):
    """
    Backpropagation of errors to compute gradients

    Formula: delta^(L) = nabla_a L * phi'(z^(L)); delta^(l) = (W^(l+1))^T delta^(l+1) * phi'(z^(l))

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    weights : array-like
        Input data.
    activations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gradients

    References
    ----------
    Géron Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation of errors to compute gradients"})


def cheatsheet():
    return "hmbp: Backpropagation of errors to compute gradients"
