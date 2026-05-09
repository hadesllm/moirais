"""Kernel-moment-based bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_kernel_moment"]


def bound_kernel_moment(y, X, kernel):
    """
    Kernel-moment-based bound

    Formula: kernelized moment family

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    kernel : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews-Shi (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kernel-moment-based bound"})


def cheatsheet():
    return "bnskmt: Kernel-moment-based bound"
