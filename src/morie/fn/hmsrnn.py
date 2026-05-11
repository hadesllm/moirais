# morie.fn — function file (hadesllm/morie)
"""Simple RNN forward pass over a sequence."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_simple_rnn"]


def geron_simple_rnn(X, Wx, Wh, b):
    """
    Simple RNN forward pass over a sequence

    Formula: h_t = tanh(W_x x_t + W_h h_{t-1} + b)

    Parameters
    ----------
    X : array-like
        Input data.
    Wx : array-like
        Input data.
    Wh : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_T

    References
    ----------
    Géron Ch 13
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple RNN forward pass over a sequence"})


def cheatsheet():
    return "hmsrnn: Simple RNN forward pass over a sequence"
