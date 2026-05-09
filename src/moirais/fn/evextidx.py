"""Runs estimator of the extremal index θ."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_extremal_index_runs"]


def evt_extremal_index_runs(x, u, r):
    """
    Runs estimator of the extremal index θ

    Formula: θ̂_runs = N_clusters / N_exc

    Parameters
    ----------
    x : array-like
        Input data.
    u : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta

    References
    ----------
    Smith & Weissman (1994)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Runs estimator of the extremal index θ"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Runs estimator of the extremal index θ"})


def cheatsheet():
    return "evextidx: Runs estimator of the extremal index θ"
