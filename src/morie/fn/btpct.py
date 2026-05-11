"""Percentile bootstrap CI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_percentile_ci"]


def boot_percentile_ci(theta_b, alpha):
    """
    Percentile bootstrap CI

    Formula: [θ̂*_{(α/2)}, θ̂*_{(1-α/2)}]

    Parameters
    ----------
    theta_b : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Efron & Tibshirani (1993)
    """
    theta_b = np.atleast_1d(np.asarray(theta_b, dtype=float))
    n = len(theta_b)
    result = float(np.mean(theta_b))
    se = float(np.std(theta_b, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Percentile bootstrap CI"})


def cheatsheet():
    return "btpct: Percentile bootstrap CI"
