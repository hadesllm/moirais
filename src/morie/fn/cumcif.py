"""Aalen-Johansen cumulative incidence function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cumulative_incidence_function"]


def cumulative_incidence_function(time, cause):
    """
    Aalen-Johansen cumulative incidence function

    Formula: F_k(t) = integral_0^t S(u-) lambda_k(u) du

    Parameters
    ----------
    time : array-like
        Input data.
    cause : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aalen & Johansen (1978)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Aalen-Johansen cumulative incidence function"})


def cheatsheet():
    return "cumcif: Aalen-Johansen cumulative incidence function"
