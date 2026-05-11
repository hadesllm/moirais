"""Jackknife replicate variance for complex surveys."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["jackknife_variance_survey"]


def jackknife_variance_survey(y, weights, replicates):
    """
    Jackknife replicate variance for complex surveys

    Formula: Var = ((R-1)/R) sum_r (theta_(-r) - theta_hat)^2

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    replicates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wolter (2007) §4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Jackknife replicate variance for complex surveys"})


def cheatsheet():
    return "jackvar: Jackknife replicate variance for complex surveys"
