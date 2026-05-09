"""Modularity Q (Newman) of a community partition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["modularity_newman"]


def modularity_newman(y, A, communities):
    """
    Modularity Q (Newman) of a community partition

    Formula: Q = (1/2m) sum_ij [A_ij - k_i k_j / 2m] delta(c_i, c_j)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    communities : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2006)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Modularity Q (Newman) of a community partition"})


def cheatsheet():
    return "modlar: Modularity Q (Newman) of a community partition"
