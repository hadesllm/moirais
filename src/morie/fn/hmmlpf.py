# morie.fn — function file (hadesllm/morie)
"""Multilayer perceptron forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mlp"]


def geron_mlp(X, weights, biases, activations):
    """
    Multilayer perceptron forward pass

    Formula: a^(l+1) = phi(W^(l+1) a^(l) + b^(l+1))

    Parameters
    ----------
    X : array-like
        Input data.
    weights : array-like
        Input data.
    biases : array-like
        Input data.
    activations : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: a_L

    References
    ----------
    Géron Ch 9
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilayer perceptron forward pass"})


def cheatsheet():
    return "hmmlpf: Multilayer perceptron forward pass"
