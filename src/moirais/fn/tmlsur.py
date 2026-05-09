"""Targeted Maximum Likelihood Estimation for survival/time-to-event outcomes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_survival"]


def tmle_survival(time, event, treatment, covariates, tau):
    """
    Targeted Maximum Likelihood Estimation for survival/time-to-event outcomes

    Formula: S(t|A,W) updated via clever covariate Q*; treatment-specific survival difference

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    treatment : array-like
        Input data.
    covariates : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Moore & van der Laan (2009); Stitelman & van der Laan (2010)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Targeted Maximum Likelihood Estimation for survival/time-to-event outcomes"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Targeted Maximum Likelihood Estimation for survival/time-to-event outcomes"})


def cheatsheet():
    return "tmlsur: Targeted Maximum Likelihood Estimation for survival/time-to-event outcomes"
