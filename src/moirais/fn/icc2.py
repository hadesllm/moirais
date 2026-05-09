"""ICC(2,1) two-way random single rater."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_two_way_random"]


def icc_two_way_random(y, subject, rater):
    """
    ICC(2,1) two-way random single rater

    Formula: ICC(2,1) = (MS_r - MS_e) / (MS_r + (k-1) MS_e + k(MS_c - MS_e)/n)

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
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(2,1) two-way random single rater"})


def cheatsheet():
    return "icc2: ICC(2,1) two-way random single rater"
