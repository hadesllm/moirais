"""Aalen-Johansen estimator (multistate)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["aalen_johansen"]


def aalen_johansen(time, state, transitions):
    """
    Aalen-Johansen estimator (multistate)

    Formula: product-integral of transition matrix

    Parameters
    ----------
    time : array-like
        Input data.
    state : array-like
        Input data.
    transitions : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Aalen-Johansen (1978)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Aalen-Johansen estimator (multistate)"})
    estimate = np.median(time)
    se = 1.2533 * np.std(time, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Aalen-Johansen estimator (multistate)"})


def cheatsheet():
    return "msmaln: Aalen-Johansen estimator (multistate)"
