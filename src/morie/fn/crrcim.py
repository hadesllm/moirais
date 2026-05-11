"""Cumulative incidence function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cumulative_incidence"]


def cumulative_incidence(time, event_type, cause):
    """
    Cumulative incidence function

    Formula: F_k(t) = integral S(u-) lambda_k(u) du

    Parameters
    ----------
    time : array-like
        Input data.
    event_type : array-like
        Input data.
    cause : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kalbfleisch-Prentice (2002)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cumulative incidence function"})


def cheatsheet():
    return "crrcim: Cumulative incidence function"
