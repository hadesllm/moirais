"""Linear min-max bound."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_linear_min_max"]


def bound_linear_min_max(theta, moments):
    """
    Linear min-max bound

    Formula: theta in [min m(theta), max m(theta)]

    Parameters
    ----------
    theta : array-like
        Input data.
    moments : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Chernozhukov-Lee-Rosen (2013)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Linear min-max bound"})


def cheatsheet():
    return "bndlmm: Linear min-max bound"
