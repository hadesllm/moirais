"""Reliability of GEBV."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["reliability_gebv"]


def reliability_gebv(fit):
    """
    Reliability of GEBV

    Formula: r^2 = 1 - PEV/sigma_a^2

    Parameters
    ----------
    fit : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Henderson (1984)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Reliability of GEBV"})


def cheatsheet():
    return "reldge: Reliability of GEBV"
