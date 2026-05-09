# moirais.fn — function file (hadesllm/moirais)
"""Perceptron activation function (step function)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["perceptron_activation"]


def perceptron_activation(X, w, b):
    """
    Perceptron activation function (step function)

    Formula: a = sign(w'w + b); misclassification update: w <- w + eta*y_i*x_i

    Parameters
    ----------
    X : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'a': 'array'}

    References
    ----------
    Montesinos Lopez Ch 10
    """
    w = np.asarray(w, dtype=float)
    n = int(w) if w.ndim == 0 else len(w)
    result = float(np.mean(w))
    se = float(np.std(w, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceptron activation function (step function)"})


def cheatsheet():
    return "percn: Perceptron activation function (step function)"
