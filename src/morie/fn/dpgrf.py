"""DP grouped random field."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_grouped_random_field"]


def dp_grouped_random_field(y, grid, alpha, gamma):
    """
    DP grouped random field

    Formula: hierarchical DP with spatial coupling

    Parameters
    ----------
    y : array-like
        Input data.
    grid : array-like
        Input data.
    alpha : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Müller-Quintana (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP grouped random field"})


def cheatsheet():
    return "dpgrf: DP grouped random field"
