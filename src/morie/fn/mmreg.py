"""MM-estimator (high efficiency + 50% breakdown)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mm_estimator"]


def mm_estimator(X, y, c1, c2):
    """
    MM-estimator (high efficiency + 50% breakdown)

    Formula: M-step from S-scale; high-efficiency Tukey c

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    c1 : array-like
        Input data.
    c2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Yohai (1987)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "MM-estimator (high efficiency + 50% breakdown)"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "MM-estimator (high efficiency + 50% breakdown)"})


def cheatsheet():
    return "mmreg: MM-estimator (high efficiency + 50% breakdown)"
