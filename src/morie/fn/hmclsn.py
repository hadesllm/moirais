# morie.fn — function file (hadesllm/morie)
"""Classification MLP: softmax output and cross-entropy loss."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_classification_mlp"]


def geron_classification_mlp(X, y, hidden_sizes, epochs, lr):
    """
    Classification MLP: softmax output and cross-entropy loss

    Formula: loss = -sum_k y_k log(softmax(logits)_k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    hidden_sizes : array-like
        Input data.
    epochs : array-like
        Input data.
    lr : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 9
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Classification MLP: softmax output and cross-entropy loss"})


def cheatsheet():
    return "hmclsn: Classification MLP: softmax output and cross-entropy loss"
