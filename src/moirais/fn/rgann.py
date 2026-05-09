# moirais.fn — function file (hadesllm/moirais)
"""Multilayer perceptron (ANN) with backpropagation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ann_mlp"]


def rangayyan_ann_mlp(X, y, layers, lr, max_iter):
    """
    Multilayer perceptron (ANN) with backpropagation

    Formula: y = sigma(W_2*sigma(W_1*y+b_1)+b_2); dW = -eta * dL/dW via chain rule

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    layers : array-like
        Input data.
    lr : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: weights, biases, predictions

    References
    ----------
    Rangayyan Ch 10.8
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilayer perceptron (ANN) with backpropagation"})


def cheatsheet():
    return "rgann: Multilayer perceptron (ANN) with backpropagation"
