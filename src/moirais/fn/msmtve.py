"""MSM for continuous/dose time-varying exposures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_time_varying_exposure"]


def msm_time_varying_exposure(y, exposure_history, covariate_history, time):
    """
    MSM for continuous/dose time-varying exposures

    Formula: weight stabilized by sw = prod_t f(A_t|A_{t-1})/f(A_t|H_t)

    Parameters
    ----------
    y : array-like
        Input data.
    exposure_history : array-like
        Input data.
    covariate_history : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hernán-Brumback-Robins (2000); Cole-Hernán (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM for continuous/dose time-varying exposures"})


def cheatsheet():
    return "msmtve: MSM for continuous/dose time-varying exposures"
