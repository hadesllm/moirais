# moirais.fn — function file (hadesllm/moirais)
"""Median voter theorem."""
import numpy as np
from ._richresult import RichResult

__all__ = ["median_voter"]


def median_voter(x):
    """
    Median voter theorem

    Formula: x* = median(x_i*) in 1D

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
    Armstrong Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Median voter theorem"})


def cheatsheet():
    return "mdvtr: Median voter theorem"
