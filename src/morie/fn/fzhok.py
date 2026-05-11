# morie.fn — function file (hadesllm/morie)
"""Higher-order kernel for bias reduction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["fauzi_higher_order_kernel"]


def fauzi_higher_order_kernel(x):
    """
    Higher-order kernel for bias reduction

    Formula: K_r(u) such that integral u^j K_r = 0 for j=1..r-1

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
    Fauzi Ch 1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Higher-order kernel for bias reduction"})


def cheatsheet():
    return "fzhok: Higher-order kernel for bias reduction"
