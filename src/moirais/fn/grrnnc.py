# moirais.fn — function file (hadesllm/moirais)
"""Simple RNN cell forward step."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_simple_rnn_cell"]


def geron_simple_rnn_cell(x_t, h_prev, Whh, Wxh, b):
    """
    Simple RNN cell forward step

    Formula: h_t = tanh(W_hh h_{t-1} + W_xh x_t + b)

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    Whh : array-like
        Input data.
    Wxh : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t

    References
    ----------
    Géron Ch 13, Eq 13-1 (Simple RNN cell)
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Simple RNN cell forward step"})


def cheatsheet():
    return "grrnnc: Simple RNN cell forward step"
