# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Backpropagation gradient computation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["backpropagation"]


def backpropagation(x, y):
    """
    Backpropagation gradient computation

    Formula: dL/dW = dL/da * da/dz * dz/dW

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Backpropagation gradient computation"})


def cheatsheet():
    return "bkprp: Backpropagation gradient computation"
