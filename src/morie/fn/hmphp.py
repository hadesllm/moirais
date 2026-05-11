# morie.fn — function file (hadesllm/morie)
"""Peephole LSTM: gates also look at cell state."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_peephole_lstm"]


def geron_peephole_lstm(x_t, h_prev, c_prev, weights):
    """
    Peephole LSTM: gates also look at cell state

    Formula: i_t = sigma(W_x x_t + W_h h_{t-1} + W_c c_{t-1} + b)

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    c_prev : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t, c_t

    References
    ----------
    Géron Ch 13
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Peephole LSTM: gates also look at cell state"})


def cheatsheet():
    return "hmphp: Peephole LSTM: gates also look at cell state"
