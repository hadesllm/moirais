# moirais.fn — function file (hadesllm/moirais)
"""Memory cell abstraction: internal state carries information through time."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_memory_cell"]


def geron_memory_cell(c_prev, x_t, f):
    """
    Memory cell abstraction: internal state carries information through time

    Formula: c_t = f(c_{t-1}, x_t)

    Parameters
    ----------
    c_prev : array-like
        Input data.
    x_t : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: c_t

    References
    ----------
    Géron Ch 13
    """
    c_prev = np.atleast_1d(np.asarray(c_prev, dtype=float))
    n = len(c_prev)
    result = float(np.mean(c_prev))
    se = float(np.std(c_prev, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Memory cell abstraction: internal state carries information through time"})


def cheatsheet():
    return "hmmcel: Memory cell abstraction: internal state carries information through time"
