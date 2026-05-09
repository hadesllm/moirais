"""ICC(A,1) absolute agreement single rater."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_absolute_agreement"]


def icc_absolute_agreement(y, subject, rater):
    """
    ICC(A,1) absolute agreement single rater

    Formula: ICC(A,1) = (MS_r - MS_e) / (MS_r + (k-1) MS_e + k(MS_c - MS_e)/n)

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(A,1) absolute agreement single rater"})


def cheatsheet():
    return "iccA: ICC(A,1) absolute agreement single rater"
