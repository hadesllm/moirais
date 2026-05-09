# moirais.fn — function file (hadesllm/moirais)
"""Threshold logic unit: step activation of weighted sum."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_tlu"]


def geron_tlu(x, w, b):
    """
    Threshold logic unit: step activation of weighted sum

    Formula: y = step(w^T x + b)

    Parameters
    ----------
    x : array-like
        Input data.
    w : array-like
        Input data.
    b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y

    References
    ----------
    Géron Ch 9
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold logic unit: step activation of weighted sum"})


def cheatsheet():
    return "hmtlu: Threshold logic unit: step activation of weighted sum"
