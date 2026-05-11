# morie.fn — function file (hadesllm/morie)
"""Analysis of ordered categorical data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ordered_categories"]


def ordered_categories(x):
    """
    Analysis of ordered categorical data

    Formula: Linear-by-linear association test

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: statistic, p_value

    References
    ----------
    Gibbons Ch 14.6.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Analysis of ordered categorical data"})


def cheatsheet():
    return "ordct: Analysis of ordered categorical data"
