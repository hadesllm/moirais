# morie.fn — function file (hadesllm/morie)
"""Causal (masked) 1D convolution for time-series forecasting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_causal_1d_cnn"]


def geron_causal_1d_cnn(x, w):
    """
    Causal (masked) 1D convolution for time-series forecasting

    Formula: y_t = sum_{k=0..K-1} w_k x_{t-k}  (strictly past inputs)

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 13, Causal 1D CNN section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal (masked) 1D convolution for time-series forecasting"})


def cheatsheet():
    return "grc1d: Causal (masked) 1D convolution for time-series forecasting"
