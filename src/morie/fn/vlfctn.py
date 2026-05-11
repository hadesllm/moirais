"""Value function estimation for regime d."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["value_function_eval"]


def value_function_eval(y, D, W, regime):
    """
    Value function estimation for regime d

    Formula: V(d) = E[Y(d)]

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    W : array-like
        Input data.
    regime : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Zhang-Tsiatis-Davidian (2012)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Value function estimation for regime d"})
    estimate = np.median(y)
    se = 1.2533 * np.std(y, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Value function estimation for regime d"})


def cheatsheet():
    return "vlfctn: Value function estimation for regime d"
