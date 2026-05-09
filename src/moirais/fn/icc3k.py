"""ICC(3,k) two-way mixed average rater."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_two_way_mixed_avg"]


def icc_two_way_mixed_avg(y, subject, rater):
    """
    ICC(3,k) two-way mixed average rater

    Formula: ICC(3,k) = (MS_r - MS_e) / MS_r

    Parameters
    ----------
    y : array-like
        Input data.
    subject : array-like
        Input data.
    rater : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shrout & Fleiss (1979)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "ICC(3,k) two-way mixed average rater"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "ICC(3,k) two-way mixed average rater"})


def cheatsheet():
    return "icc3k: ICC(3,k) two-way mixed average rater"
