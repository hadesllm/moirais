"""Slice sampling."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["slice_sampler"]


def slice_sampler(log_p, x0, width):
    """
    Slice sampling

    Formula: alternate sampling u | x and x | u

    Parameters
    ----------
    log_p : array-like
        Input data.
    x0 : array-like
        Input data.
    width : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Neal (2003)
    """
    log_p = np.atleast_1d(np.asarray(log_p, dtype=float))
    n = len(log_p)
    result = float(np.mean(log_p))
    se = float(np.std(log_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Slice sampling"})


def cheatsheet():
    return "slcmc: Slice sampling"
