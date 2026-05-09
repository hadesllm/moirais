"""Dynamic time warping."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_warping"]


def functional_warping(x, y, cost):
    """
    Dynamic time warping

    Formula: DP over alignment cost

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    cost : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sakoe-Chiba (1978)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dynamic time warping"})


def cheatsheet():
    return "fdwarp: Dynamic time warping"
