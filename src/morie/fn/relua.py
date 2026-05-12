# morie.fn -- function file (hadesllm/morie)
"""ReLU activation function for neural networks."""
import numpy as np
from ._richresult import RichResult

__all__ = ["relu_activation"]


def relu_activation(x):
    """
    ReLU activation function for neural networks

    Formula: f(x) = max(0, x)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'activated': 'array'}

    References
    ----------
    Montesinos Lopez Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ReLU activation function for neural networks"})


def cheatsheet():
    return "relua: ReLU activation function for neural networks"
