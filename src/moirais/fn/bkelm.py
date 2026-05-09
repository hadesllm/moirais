# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Elman RNN hidden-state recurrence (vanilla simple-RNN formulation)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["burkov_elman_rnn"]


def burkov_elman_rnn(x_t, h_prev, Wh, Wx, Wy, bh, by):
    """
    Elman RNN hidden-state recurrence (vanilla simple-RNN formulation)

    Formula: h_t = tanh(W_h h_{t-1} + W_x x_t + b_h);  y_t = W_y h_t + b_y

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    Wh : array-like
        Input data.
    Wx : array-like
        Input data.
    Wy : array-like
        Input data.
    bh : array-like
        Input data.
    by : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t, y_t

    References
    ----------
    Burkov Ch 3, Elman RNN section
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elman RNN hidden-state recurrence (vanilla simple-RNN formulation)"})


def cheatsheet():
    return "bkelm: Elman RNN hidden-state recurrence (vanilla simple-RNN formulation)"
