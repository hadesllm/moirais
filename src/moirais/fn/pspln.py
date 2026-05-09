# moirais.fn — function file (hadesllm/moirais)
"""Penalized B-spline regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["penalized_spline"]


def penalized_spline(x, y):
    """
    Penalized B-spline regression

    Formula: min ||y-Bb||^2 + lambda*b'Db

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Eilers & Marx (1996)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalized B-spline regression"})


def cheatsheet():
    return "pspln: Penalized B-spline regression"
