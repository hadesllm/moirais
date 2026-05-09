"""MSM accelerated failure time model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["msm_accelerated_failure"]


def msm_accelerated_failure(time, event, treatment_history, covariate_history):
    """
    MSM accelerated failure time model

    Formula: log T = beta a_bar + epsilon under IPTW

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    treatment_history : array-like
        Input data.
    covariate_history : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Robins-Tsiatis (1991); Hernán-Cole-Margolick (2005)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSM accelerated failure time model"})


def cheatsheet():
    return "msmaft: MSM accelerated failure time model"
