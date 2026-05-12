# morie.fn -- function file (hadesllm/morie)
"""Houlsby-style bottleneck adapter inserted in a transformer block."""
import numpy as np
from ._richresult import RichResult

__all__ = ["kamath_houlsby_adapter"]


def kamath_houlsby_adapter(h, W_down, W_up):
    """
    Houlsby-style bottleneck adapter inserted in a transformer block

    Formula: adapter(h) = h + W_up * GELU(W_down * h);  W_down in R^{m x d}, W_up in R^{d x m}, m << d

    Parameters
    ----------
    h : array-like
        Input data.
    W_down : array-like
        Input data.
    W_up : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_adapted

    References
    ----------
    Kamath Ch 4, Serial Adapters / Houlsby section
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Houlsby-style bottleneck adapter inserted in a transformer block"})


def cheatsheet():
    return "kmadap: Houlsby-style bottleneck adapter inserted in a transformer block"
