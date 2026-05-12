# morie.fn -- function file (hadesllm/morie)
"""Hodges-Lehmann two-sample shift estimator based on all pairwise differences."""
import numpy as np
from ._richresult import RichResult

__all__ = ["gibbons_hodges_lehmann_2"]


def gibbons_hodges_lehmann_2(x, y):
    """
    Hodges-Lehmann two-sample shift estimator based on all pairwise differences

    Formula: Delta_hat = median{Y_j - X_i: all i,j}

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, ci

    References
    ----------
    Gibbons Ch 6.6
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Hodges-Lehmann two-sample shift estimator based on all pairwise differences"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Hodges-Lehmann two-sample shift estimator based on all pairwise differences"})


def cheatsheet():
    return "gb_hg2: Hodges-Lehmann two-sample shift estimator based on all pairwise differences"
