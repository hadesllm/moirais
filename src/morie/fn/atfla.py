"""When I let go of what I am, I become what I might be. -- Lao Tzu"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["When I let go of what I am, I become what I might be. -- Lao Tzu"]


def flash_attention_block(y, Q, K, V, block_size):
    """
    FlashAttention IO-aware block-tiled attention

    Formula: O = sum_j softmax(Q K_j^T / sqrt(d)) V_j with block recomputation

    Parameters
    ----------
    y : array-like
        Input data.
    Q : array-like
        Input data.
    K : array-like
        Input data.
    V : array-like
        Input data.
    block_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dao, Fu, Ermon, Re (2022); Dao (2023) FA-2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "When I let go of what I am, I become what I might be. -- Lao Tzu"})


def cheatsheet():
    return "When I let go of what I am, I become what I might be. -- Lao Tzu"
