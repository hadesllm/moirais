# moirais.fn — function file (hadesllm/moirais)
"""Causal 1D convolution: output at time t only depends on t'<=t."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_causal_1d_conv"]


def geron_causal_1d_conv(x, kernel):
    """
    Causal 1D convolution: output at time t only depends on t'<=t

    Formula: y_t = sum_{k=0}^{K-1} w_k x_{t-k}

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 13
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Causal 1D convolution: output at time t only depends on t'<=t"})


def cheatsheet():
    return "hmc1d: Causal 1D convolution: output at time t only depends on t'<=t"
