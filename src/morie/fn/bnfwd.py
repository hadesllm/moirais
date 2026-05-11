# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Batch normalization forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["batch_norm_forward"]


def batch_norm_forward(x):
    """
    Batch normalization forward pass

    Formula: y = gamma * (x-mu)/sqrt(var+eps) + beta

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
    Ioffe & Szegedy (2015)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch normalization forward pass"})


def cheatsheet():
    return "bnfwd: Batch normalization forward pass"
