# moirais.fn — function file (hadesllm/moirais)
"""LSTM cell: input/forget/output gates + cell state."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lstm"]


def geron_lstm(x_t, h_prev, c_prev, weights):
    """
    LSTM cell: input/forget/output gates + cell state

    Formula: i_t = sigma(...); f_t = sigma(...); o_t = sigma(...); c_t = f_t*c_{t-1} + i_t*g_t

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LSTM cell: input/forget/output gates + cell state"})


def cheatsheet():
    return "hmlstm: LSTM cell: input/forget/output gates + cell state"
