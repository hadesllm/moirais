"""Proportion of total effect mediated."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proportion_mediated"]


def proportion_mediated(a, b, c_prime):
    """
    Proportion of total effect mediated

    Formula: PM = (a*b) / (a*b + c')

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    c_prime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    MacKinnon (2008) §4
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportion of total effect mediated"})


def cheatsheet():
    return "propme: Proportion of total effect mediated"
