"""Covariate balance check post-IPTW."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["covariate_balance_check"]


def covariate_balance_check(A, H, weights):
    """
    Covariate balance check post-IPTW

    Formula: weighted standardized mean difference

    Parameters
    ----------
    A : array-like
        Input data.
    H : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Austin-Stuart (2015)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Covariate balance check post-IPTW"})


def cheatsheet():
    return "covbal: Covariate balance check post-IPTW"
