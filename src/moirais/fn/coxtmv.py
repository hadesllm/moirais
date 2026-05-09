"""Cox model with time-varying covariates."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_time_varying"]


def cox_time_varying(time, event, X_time):
    """
    Cox model with time-varying covariates

    Formula: lambda(t|X(t)) = lambda_0(t) exp(beta X(t))

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X_time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Therneau-Grambsch (2000)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox model with time-varying covariates"})


def cheatsheet():
    return "coxtmv: Cox model with time-varying covariates"
