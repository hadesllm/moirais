"""ICC(3,1) two-way mixed single rater (consistency)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_two_way_mixed"]


def icc_two_way_mixed(y, subject, rater):
    """
    ICC(3,1) two-way mixed single rater (consistency)

    Formula: ICC(3,1) = (MS_r - MS_e) / (MS_r + (k-1) MS_e)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(3,1) two-way mixed single rater (consistency)"})


def cheatsheet():
    return "icc3: ICC(3,1) two-way mixed single rater (consistency)"
