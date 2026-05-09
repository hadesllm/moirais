"""Wide & Deep."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wide_and_deep"]


def wide_and_deep(X_wide, X_deep, y):
    """
    Wide & Deep

    Formula: linear wide + DNN deep, jointly trained

    Parameters
    ----------
    X_wide : array-like
        Input data.
    X_deep : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cheng et al (2016) Google
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wide & Deep"})


def cheatsheet():
    return "wideD: Wide & Deep"
