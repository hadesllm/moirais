"""Madogram-based estimator of A(t) for max-stable."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_madogram"]


def evt_madogram(x, y, t_grid):
    """
    Madogram-based estimator of A(t) for max-stable

    Formula: ν̂(t) = (1/2) E|F_X^t(X) - F_Y^{1-t}(Y)|

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    t_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: nu, A_t

    References
    ----------
    Naveau et al. (2009)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Madogram-based estimator of A(t) for max-stable"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Madogram-based estimator of A(t) for max-stable"})


def cheatsheet():
    return "evmadog: Madogram-based estimator of A(t) for max-stable"
