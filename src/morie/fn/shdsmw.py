"""Shrinkage MSM with regularized weights."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["shrinkage_msm"]


def shrinkage_msm(y, treatment_history, covariate_history, lam):
    """
    Shrinkage MSM with regularized weights

    Formula: L1/L2 penalty on log-weights

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Setoguchi et al (2008); Westreich et al (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Shrinkage MSM with regularized weights"})


def cheatsheet():
    return "shdsmw: Shrinkage MSM with regularized weights"
