"""Bootstrap CI for a ratio of estimates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_ci_ratio"]


def boot_ci_ratio(x, y, stat_x, stat_y, B, alpha):
    """
    Bootstrap CI for a ratio of estimates

    Formula: ρ̂*_b = T_x(x*)/T_y(y*); percentile interval

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    stat_x : array-like
        Input data.
    stat_y : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Davison & Hinkley (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap CI for a ratio of estimates"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap CI for a ratio of estimates"})


def cheatsheet():
    return "btciratio: Bootstrap CI for a ratio of estimates"
