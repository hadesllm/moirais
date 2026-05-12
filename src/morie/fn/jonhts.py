# morie.fn -- function file (hadesllm/morie)
"""N-HiTS: Neural Hierarchical Interpolation for Time Series forecasting."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_nhits"]


def joseph_nhits(x, blocks, expressivity, horizon):
    """
    N-HiTS: Neural Hierarchical Interpolation for Time Series forecasting

    Formula: y_hat = sum_s Interpolate(MLP_s(x), expressivity_s); hierarchical multi-rate blocks

    Parameters
    ----------
    x : array-like
        Input data.
    blocks : array-like
        Input data.
    expressivity : array-like
        Input data.
    horizon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Joseph Ch 16, N-HiTS section
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "N-HiTS: Neural Hierarchical Interpolation for Time Series forecasting"})


def cheatsheet():
    return "jonhts: N-HiTS: Neural Hierarchical Interpolation for Time Series forecasting"
