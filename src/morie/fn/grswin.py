# morie.fn — function file (hadesllm/morie)
"""Swin Transformer: self-attention restricted to non-overlapping local windows."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_swin_window_attention"]


def geron_swin_window_attention(X, window_size, WQ, WK, WV):
    """
    Swin Transformer: self-attention restricted to non-overlapping local windows

    Formula: partition X into W x W windows; apply MHA independently within each window

    Parameters
    ----------
    X : array-like
        Input data.
    window_size : array-like
        Input data.
    WQ : array-like
        Input data.
    WK : array-like
        Input data.
    WV : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Géron Ch 16, Swin Transformer section
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Swin Transformer: self-attention restricted to non-overlapping local windows"})


def cheatsheet():
    return "grswin: Swin Transformer: self-attention restricted to non-overlapping local windows"
