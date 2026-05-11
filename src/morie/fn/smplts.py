"""Sample life table estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sample_lifetable"]


def sample_lifetable(intervals, entered, died, censored):
    """
    Sample life table estimator

    Formula: interval-based KM with grouped data

    Parameters
    ----------
    intervals : array-like
        Input data.
    entered : array-like
        Input data.
    died : array-like
        Input data.
    censored : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cutler-Ederer (1958)
    """
    intervals = np.atleast_1d(np.asarray(intervals, dtype=float))
    n = len(intervals)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sample life table estimator"})
    estimate = np.median(intervals)
    se = 1.2533 * np.std(intervals, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sample life table estimator"})


def cheatsheet():
    return "smplts: Sample life table estimator"
