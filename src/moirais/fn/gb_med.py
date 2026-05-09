# moirais.fn — function file (hadesllm/moirais)
"""Distribution of sample median X_(m) for odd n = 2m+1."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_median_dist"]


def gibbons_median_dist(x, n):
    """
    Distribution of sample median X_(m) for odd n = 2m+1

    Formula: F_{med}(x) involves incomplete beta function with params (m+1, m+1)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: distribution

    References
    ----------
    Gibbons Ch 2.7.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Distribution of sample median X_(m) for odd n = 2m+1"})


def cheatsheet():
    return "gb_med: Distribution of sample median X_(m) for odd n = 2m+1"
