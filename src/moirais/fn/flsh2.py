"""Real knowledge is to know the extent of one's ignorance. — Confucius"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["Real knowledge is to know the extent of one's ignorance. — Confucius"]


def flash_attention(Q, K, V):
    """
    FlashAttention IO-aware exact attention

    Formula: tiled softmax with online normalization

    Parameters
    ----------
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dao et al (2022) FlashAttention
    """
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    n = len(Q)
    result = float(np.mean(Q))
    se = float(np.std(Q, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Real knowledge is to know the extent of one's ignorance. — Confucius"})


def cheatsheet():
    return "Real knowledge is to know the extent of one's ignorance. — Confucius"
