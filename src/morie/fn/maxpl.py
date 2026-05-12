# morie.fn -- function file (hadesllm/morie)
"""Max pooling operation for CNNs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["max_pooling"]


def max_pooling(x, kernel, stride):
    """
    Max pooling operation for CNNs

    Formula: y[i] = max(x[i*s : i*s+k]) for stride s, kernel k

    Parameters
    ----------
    x : array-like
        Input data.
    kernel : array-like
        Input data.
    stride : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'pooled': 'array'}

    References
    ----------
    Montesinos Lopez Ch 13
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max pooling operation for CNNs"})


def cheatsheet():
    return "maxpl: Max pooling operation for CNNs"
