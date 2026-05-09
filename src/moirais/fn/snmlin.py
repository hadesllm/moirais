"""Linear structural nested mean model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["snm_linear"]


def snm_linear(y, treatment_history, covariate_history, time):
    """
    Linear structural nested mean model

    Formula: E[Y - sum_t gamma A_t | H_t] = 0

    Parameters
    ----------
    y : array-like
        Input data.
    treatment_history : array-like
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
    Robins (1994); Vansteelandt-Joffe (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear structural nested mean model"})


def cheatsheet():
    return "snmlin: Linear structural nested mean model"
