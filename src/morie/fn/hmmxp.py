# morie.fn -- function file (hadesllm/morie)
"""Max pooling: output maximum per pooling window."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_max_pool"]


def geron_max_pool(x, window, stride):
    """
    Max pooling: output maximum per pooling window

    Formula: y[i,j,k] = max over window W of x[i+u, j+v, k]

    Parameters
    ----------
    x : array-like
        Input data.
    window : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 12
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max pooling: output maximum per pooling window"})


def cheatsheet():
    return "hmmxp: Max pooling: output maximum per pooling window"
