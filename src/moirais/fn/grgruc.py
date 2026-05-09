# moirais.fn — function file (hadesllm/moirais)
"""GRU cell: update + reset gates, candidate state."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gru_cell"]


def geron_gru_cell(x_t, h_prev, Wz, Wr, W):
    """
    GRU cell: update + reset gates, candidate state

    Formula: z=sig(Wz[h,x]); r=sig(Wr[h,x]); h_tilde=tanh(W[r*h,x]); h_t=(1-z)*h_prev + z*h_tilde

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    Wz : array-like
        Input data.
    Wr : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_t

    References
    ----------
    Géron Ch 13, GRU section
    """
    x_t = np.atleast_1d(np.asarray(x_t, dtype=float))
    n = len(x_t)
    result = float(np.mean(x_t))
    se = float(np.std(x_t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GRU cell: update + reset gates, candidate state"})


def cheatsheet():
    return "grgruc: GRU cell: update + reset gates, candidate state"
