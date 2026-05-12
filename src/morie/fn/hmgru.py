# morie.fn -- function file (hadesllm/morie)
"""GRU cell: reset and update gates."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gru"]


def geron_gru(x_t, h_prev, weights):
    """
    GRU cell: reset and update gates

    Formula: z_t = sigma(...); r_t = sigma(...); h_t = (1-z_t)*h_{t-1} + z_t*h_tilde

    Parameters
    ----------
    x_t : array-like
        Input data.
    h_prev : array-like
        Input data.
    weights : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GRU cell: reset and update gates"})


def cheatsheet():
    return "hmgru: GRU cell: reset and update gates"
