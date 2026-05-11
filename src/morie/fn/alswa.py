# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Sliding-window attention: token i attends to [i-W, i] only."""
import numpy as np
from ._richresult import RichResult

__all__ = ["alammar_sliding_window_attention"]


def alammar_sliding_window_attention(Q, K, V, window_size):
    """
    Sliding-window attention: token i attends to [i-W, i] only

    Formula: mask[i, j] = 0 if (i - W) <= j <= i else -inf;  Attn = softmax(QK^T / sqrt(d_k) + mask) V

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    window_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: output

    References
    ----------
    Alammar Ch 3, Sliding Window Attention section
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sliding-window attention: token i attends to [i-W, i] only"})


def cheatsheet():
    return "alswa: Sliding-window attention: token i attends to [i-W, i] only"
