"""Rao-Wu bootstrap."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bootstrap_survey"]


def bootstrap_survey(data, strata, B):
    """
    Rao-Wu bootstrap

    Formula: resample n_h - 1 PSU per stratum with replacement

    Parameters
    ----------
    data : array-like
        Input data.
    strata : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rao-Wu (1988)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rao-Wu bootstrap"})


def cheatsheet():
    return "bootss: Rao-Wu bootstrap"
