"""ICC(C,1) consistency single rater."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_consistency"]


def icc_consistency(y, subject, rater):
    """
    ICC(C,1) consistency single rater

    Formula: ICC(C,1) = (MS_r - MS_e) / (MS_r + (k-1) MS_e)

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
    McGraw & Wong (1996)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(C,1) consistency single rater"})


def cheatsheet():
    return "iccC: ICC(C,1) consistency single rater"
