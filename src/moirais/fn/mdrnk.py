# moirais.fn — function file (hadesllm/moirais)
"""Midrank computation for tied observations."""
import numpy as np
from ._richresult import RichResult

__all__ = ["midranks"]


def midranks(x):
    """
    Midrank computation for tied observations

    Formula: midrank = (rank_lower + rank_upper) / 2

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
    Gibbons Ch 5.6.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Midrank computation for tied observations"})


def cheatsheet():
    return "mdrnk: Midrank computation for tied observations"
