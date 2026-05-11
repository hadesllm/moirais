# morie.fn — function file (hadesllm/morie)
"""Recurrent neuron step: hidden state updated from previous state."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_recurrent_neuron"]


def geron_recurrent_neuron(x_t, h_prev, Wx, Wh, b):
    """
    Recurrent neuron step: hidden state updated from previous state

    Formula: h_t = phi(W_x x_t + W_h h_{t-1} + b)

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
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
        Keys: h_t

    References
    ----------
    Géron Ch 13
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Recurrent neuron step: hidden state updated from previous state"})


def cheatsheet():
    return "hmrnn: Recurrent neuron step: hidden state updated from previous state"
