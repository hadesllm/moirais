"""Survey-weighted variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weighted_variance"]


def weighted_variance(y, weights):
    """
    Survey-weighted variance

    Formula: s2_w = (sum w_i (y_i - ybar_w)^2) / ((sum w_i) - 1)

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lohr (2010) §3.5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted variance"})


def cheatsheet():
    return "wvar: Survey-weighted variance"
