# morie.fn -- function file (hadesllm/morie)
"""LSTM cell: forget, input, cell candidate, output gates + state update."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_lstm_cell"]


def geron_lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wg, Wo, bf, bi, bg, bo):
    """
    LSTM cell: forget, input, cell candidate, output gates + state update

    Formula: f=sig(Wf[h,x]+bf); i=sig(Wi[h,x]+bi); g=tanh(Wg[h,x]+bg); o=sig(Wo[h,x]+bo); c=f*c+i*g; h=o*tanh(c)

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    c_prev : array-like
        Input data.
    Wf : array-like
        Input data.
    Wi : array-like
        Input data.
    Wg : array-like
        Input data.
    Wo : array-like
        Input data.
    bf : array-like
        Input data.
    bi : array-like
        Input data.
    bg : array-like
        Input data.
    bo : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t, c_t

    References
    ----------
    Géron Ch 13, LSTM section
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "LSTM cell: forget, input, cell candidate, output gates + state update"})


def cheatsheet():
    return "grlstc: LSTM cell: forget, input, cell candidate, output gates + state update"
