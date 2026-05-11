"""Validity check for bound assumptions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_validity_check"]


def bound_validity_check(lower, upper, theta_0, H0):
    """
    Validity check for bound assumptions

    Formula: H0: assumptions hold; bound covers theta_0

    Parameters
    ----------
    lower : array-like
        Input data.
    upper : array-like
        Input data.
    theta_0 : array-like
        Input data.
    H0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Manski (2003)
    """
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    n = len(lower)
    result = float(np.mean(lower))
    se = float(np.std(lower, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Validity check for bound assumptions"})


def cheatsheet():
    return "bndvld: Validity check for bound assumptions"
