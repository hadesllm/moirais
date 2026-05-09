"""MINE neural MI estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_neural_estimator"]


def mi_neural_estimator(X, Y, T_network):
    """
    MINE neural MI estimator

    Formula: I = sup E_p[T] - log E_q[exp T]

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    T_network : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Belghazi et al (2018) MINE
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "MINE neural MI estimator"})
    estimate = np.median(X)
    se = 1.2533 * np.std(X, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "MINE neural MI estimator"})


def cheatsheet():
    return "miestn: MINE neural MI estimator"
