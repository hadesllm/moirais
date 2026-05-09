"""DP count."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_count"]


def dp_count(D, epsilon):
    """
    DP count

    Formula: |D| + Lap(1/ε)

    Parameters
    ----------
    D : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork et al (2006)
    """
    D = np.atleast_1d(np.asarray(D, dtype=float))
    n = len(D)
    result = float(np.mean(D))
    se = float(np.std(D, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP count"})


def cheatsheet():
    return "dpcnt: DP count"
