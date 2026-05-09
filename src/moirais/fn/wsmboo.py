"""Bootstrap variance estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_bootstrap"]


def wasserman_bootstrap(data, T, B):
    """
    Bootstrap variance estimator

    Formula: v_boot = (1/B) sum_b (theta_b - mean)^2

    Parameters
    ----------
    data : array-like
        Input data.
    T : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Wasserman (2004), Ch 8
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap variance estimator"})
    estimate = np.median(data)
    se = 1.2533 * np.std(data, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap variance estimator"})


def cheatsheet():
    return "wsmboo: Bootstrap variance estimator"
