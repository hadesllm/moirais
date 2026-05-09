"""Sen's slope estimator."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sen_slope"]


def sen_slope(x):
    """
    Sen's slope estimator

    Formula: median over (x_j - x_i)/(j - i)

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Sen (1968)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Sen's slope estimator"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Sen's slope estimator"})


def cheatsheet():
    return "sensSlp: Sen's slope estimator"
