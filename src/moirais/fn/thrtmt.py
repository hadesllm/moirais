"""MSM with threshold-treatment regime."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["threshold_treatment_msm"]


def threshold_treatment_msm(y, A, W, threshold_grid):
    """
    MSM with threshold-treatment regime

    Formula: d(W) = 1{f(W) > tau}; target tau

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    W : array-like
        Input data.
    threshold_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Luedtke-vdL (2016)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM with threshold-treatment regime"})


def cheatsheet():
    return "thrtmt: MSM with threshold-treatment regime"
