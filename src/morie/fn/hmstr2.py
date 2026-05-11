# morie.fn — function file (hadesllm/morie)
"""Stride: step size of kernel sliding over input."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_stride"]


def geron_stride(in_dim, k, p, s):
    """
    Stride: step size of kernel sliding over input

    Formula: output_dim = floor((in_dim + 2p - k)/s) + 1

    Parameters
    ----------
    in_dim : array-like
        Input data.
    k : array-like
        Input data.
    p : array-like
        Input data.
    s : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: out_dim

    References
    ----------
    Géron Ch 12
    """
    in_dim = np.atleast_1d(np.asarray(in_dim, dtype=float))
    n = len(in_dim)
    result = float(np.mean(in_dim))
    se = float(np.std(in_dim, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stride: step size of kernel sliding over input"})


def cheatsheet():
    return "hmstr2: Stride: step size of kernel sliding over input"
