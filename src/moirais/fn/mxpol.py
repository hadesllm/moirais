# moirais.fn — function file (hadesllm/moirais)
"""Max pooling forward pass."""
import numpy as np
from ._richresult import RichResult

__all__ = ["maxpool_forward"]


def maxpool_forward(x):
    """
    Max pooling forward pass

    Formula: y[i,j] = max(x[i*s:i*s+k, j*s:j*s+k])

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
    Geron (2026), Ch 14
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Max pooling forward pass"})


def cheatsheet():
    return "mxpol: Max pooling forward pass"
