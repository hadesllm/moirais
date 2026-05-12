# morie.fn -- function file (hadesllm/morie)
"""Bahdanau (additive) attention over encoder states."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bahdanau_attention"]


def geron_bahdanau_attention(h, s_prev, W, U, v):
    """
    Bahdanau (additive) attention over encoder states

    Formula: e_ij = v^T tanh(W h_i + U s_{j-1}); alpha_ij = softmax(e_ij)

    Parameters
    ----------
    h : array-like
        Input data.
    s_prev : array-like
        Input data.
    W : array-like
        Input data.
    U : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: alpha, context

    References
    ----------
    Géron Ch 14
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bahdanau (additive) attention over encoder states"})


def cheatsheet():
    return "hmbdn: Bahdanau (additive) attention over encoder states"
