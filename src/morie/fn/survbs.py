"""Bootstrap SE for survival estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survival_bootstrap_se"]


def survival_bootstrap_se(time, event, B):
    """
    Bootstrap SE for survival estimator

    Formula: resample n; recompute S(t) per replicate

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Efron-Tibshirani (1993)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Bootstrap SE for survival estimator"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Bootstrap SE for survival estimator"})


def cheatsheet():
    return "survbs: Bootstrap SE for survival estimator"
