# morie.fn — function file (hadesllm/morie)
"""Higuchi fractal dimension."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_higuchi_fd"]


def rangayyan_higuchi_fd(x):
    """
    Higuchi fractal dimension

    Formula: L(k) = (1/k) sum |x(m+ik) - x(m+(i-1)k)|

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
    Rangayyan Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Higuchi fractal dimension"})


def cheatsheet():
    return "rghfd: Higuchi fractal dimension"
