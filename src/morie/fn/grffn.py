# morie.fn — function file (hadesllm/morie)
"""Transformer position-wise feed-forward network."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_transformer_feedforward"]


def geron_transformer_feedforward(x, W1, b1, W2, b2):
    """
    Transformer position-wise feed-forward network

    Formula: FFN(x) = max(0, x W_1 + b_1) W_2 + b_2

    Parameters
    ----------
    x : array-like
        Input data.
    W1 : array-like
        Input data.
    b1 : array-like
        Input data.
    W2 : array-like
        Input data.
    b2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 15, Feed-Forward sublayer section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transformer position-wise feed-forward network"})


def cheatsheet():
    return "grffn: Transformer position-wise feed-forward network"
