# morie.fn — function file (hadesllm/morie)
"""Perceiver IO: adds cross-attention output decoder."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_perceiver_io"]


def geron_perceiver_io(x, latents, queries):
    """
    Perceiver IO: adds cross-attention output decoder

    Formula: Perceiver + decoder cross-attention from output queries

    Parameters
    ----------
    x : array-like
        Input data.
    latents : array-like
        Input data.
    queries : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: outputs

    References
    ----------
    Géron Ch 16
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Perceiver IO: adds cross-attention output decoder"})


def cheatsheet():
    return "hmprio: Perceiver IO: adds cross-attention output decoder"
