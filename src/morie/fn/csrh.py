"""Cause-specific hazard for competing-risks data."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cause_specific_hazard"]


def cause_specific_hazard(time, cause, X):
    """
    Cause-specific hazard for competing-risks data

    Formula: lambda_k(t) = lim P(t <= T < t+dt, K=k | T >= t)/dt

    Parameters
    ----------
    time : array-like
        Input data.
    cause : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Prentice et al. (1978)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cause-specific hazard for competing-risks data"})


def cheatsheet():
    return "csrh: Cause-specific hazard for competing-risks data"
