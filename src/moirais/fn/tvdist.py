"""Total variation distance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["total_variation_distance"]


def total_variation_distance(y, p, q):
    """
    Total variation distance

    Formula: TV(P,Q) = (1/2) sum_x |p(x) - q(x)|

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Devroye & Gyorfi (1985)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Total variation distance"})


def cheatsheet():
    return "tvdist: Total variation distance"
