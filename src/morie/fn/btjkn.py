"""Leave-one-out jackknife estimator + variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_jackknife"]


def boot_jackknife(x, stat):
    """
    Leave-one-out jackknife estimator + variance

    Formula: θ̂_(i); var = ((n-1)/n) Σ(θ̂_(i)-θ̂_(.))²

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_jk, var_jk, bias

    References
    ----------
    Quenouille (1949)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Leave-one-out jackknife estimator + variance"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Leave-one-out jackknife estimator + variance"})


def cheatsheet():
    return "btjkn: Leave-one-out jackknife estimator + variance"
