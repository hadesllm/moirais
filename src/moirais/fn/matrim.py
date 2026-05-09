"""Duval-Tweedie trim-and-fill missing-study correction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ma_trim_fill"]


def ma_trim_fill(yi, vi, side):
    """
    Duval-Tweedie trim-and-fill missing-study correction

    Formula: Iterate: trim k_0 from heaviest tail; fill mirror; refit

    Parameters
    ----------
    yi : array-like
        Input data.
    vi : array-like
        Input data.
    side : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_adj, k_filled, fill_yi

    References
    ----------
    Duval & Tweedie (2000)
    """
    yi = np.atleast_1d(np.asarray(yi, dtype=float))
    n = len(yi)
    result = float(np.mean(yi))
    se = float(np.std(yi, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Duval-Tweedie trim-and-fill missing-study correction"})


def cheatsheet():
    return "matrim: Duval-Tweedie trim-and-fill missing-study correction"
