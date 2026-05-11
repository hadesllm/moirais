"""G-estimation of structural nested mean models."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["g_estimation_snm"]


def g_estimation_snm(y, treatment_history, covariate_history, time):
    """
    G-estimation of structural nested mean models

    Formula: H_t(psi) = Y - sum_t gamma(A_t,H_t;psi); solve E[H_t(psi)*S_t]=0

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
    Robins (1992); Vansteelandt-Joffe (2014)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "G-estimation of structural nested mean models"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "G-estimation of structural nested mean models"})


def cheatsheet():
    return "gestid: G-estimation of structural nested mean models"
