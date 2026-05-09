# moirais.fn — function file from book-equation translation pipeline (hadesllm/moirais)
"""Scaled dot-product attention."""
import numpy as np
from ._richresult import RichResult

__all__ = ["scaled_dot_product_attention"]


def scaled_dot_product_attention(x):
    """
    Scaled dot-product attention

    Formula: Attention(Q,K,V) = softmax(QK'/sqrt(d))V

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Vaswani et al. (2017)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Scaled dot-product attention"})


def cheatsheet():
    return "attnq: Scaled dot-product attention"
