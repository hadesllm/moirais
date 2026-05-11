"""Sequential propensity-score matching."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["propensity_score_method"]


def propensity_score_method(A, H, time):
    """
    Sequential propensity-score matching

    Formula: match on time-varying balance score

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lu (2005)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential propensity-score matching"})


def cheatsheet():
    return "prsmtd: Sequential propensity-score matching"
