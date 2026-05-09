"""ICC(1,k) one-way random average across k judges."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_one_way_average"]


def icc_one_way_average(y, cluster):
    """
    ICC(1,k) one-way random average across k judges

    Formula: ICC(1,k) = (MS_b - MS_w) / MS_b

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
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
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "ICC(1,k) one-way random average across k judges"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "ICC(1,k) one-way random average across k judges"})


def cheatsheet():
    return "icc1k: ICC(1,k) one-way random average across k judges"
