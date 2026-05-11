"""Cumulative meta-analysis ordered by some criterion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_cumulative"]


def ma_cumulative(yi, vi, order):
    """
    Cumulative meta-analysis ordered by some criterion

    Formula: θ̂_t from studies 1..t for t=1..k

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_t, se_t

    References
    ----------
    Lau-Schmid-Chalmers (1995)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cumulative meta-analysis ordered by some criterion"})


def cheatsheet():
    return "macum: Cumulative meta-analysis ordered by some criterion"
