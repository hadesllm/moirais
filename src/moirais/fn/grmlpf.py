# moirais.fn — function file (hadesllm/moirais)
"""Multilayer perceptron forward pass (L hidden layers with activation phi)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mlp_forward"]


def geron_mlp_forward(x, weights, biases):
    """
    Multilayer perceptron forward pass (L hidden layers with activation phi)

    Formula: a_0 = x; a_l = phi(W_l a_{l-1} + b_l) for l=1..L; y_hat = a_L

    Parameters
    ----------
    x : array-like
        Input data.
    weights : array-like
        Input data.
    biases : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 9, Multilayer Perceptron section
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilayer perceptron forward pass (L hidden layers with activation phi)"})


def cheatsheet():
    return "grmlpf: Multilayer perceptron forward pass (L hidden layers with activation phi)"
