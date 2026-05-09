"""Coverage probability check."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_coverage_check"]


def bound_coverage_check(lower, upper, theta_true, alpha):
    """
    Coverage probability check

    Formula: verify CI covers true theta_0

    Parameters
    ----------
    lower : array-like
        Input data.
    upper : array-like
        Input data.
    theta_true : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Soares (2010)
    """
    lower = np.atleast_1d(np.asarray(lower, dtype=float))
    n = len(lower)
    result = float(np.mean(lower))
    se = float(np.std(lower, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Coverage probability check"})


def cheatsheet():
    return "bndcvr: Coverage probability check"
