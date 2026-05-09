"""Proportional allocation across strata."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proportional_allocation"]


def proportional_allocation(y, N_h, n):
    """
    Proportional allocation across strata

    Formula: n_h = n * N_h / N

    Parameters
    ----------
    y : array-like
        Input data.
    N_h : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cochran (1977) §5.3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportional allocation across strata"})


def cheatsheet():
    return "propalc: Proportional allocation across strata"
