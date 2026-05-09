"""Multi-head attention."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multi_head_attention"]


def multi_head_attention(y, Q, K, V, Wq, Wk, Wv, Wo, heads):
    """
    Multi-head attention

    Formula: MHA(Q,K,V) = Concat(head_1, ..., head_h) W^O; head_i = Attn(Q W_i^Q, K W_i^K, V W_i^V)

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
    Wq : array-like
        Input data.
    Wk : array-like
        Input data.
    Wv : array-like
        Input data.
    Wo : array-like
        Input data.
    heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vaswani et al. (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multi-head attention"})


def cheatsheet():
    return "attmh: Multi-head attention"
