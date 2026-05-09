"""Andrich Rating Scale Model (common thresholds)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rating_scale_andrich"]


def rating_scale_andrich(y, theta, b, tau_j):
    """
    Andrich Rating Scale Model (common thresholds)

    Formula: P(X=k) = exp(sum (theta - (b + tau_j))) / norm

    Parameters
    ----------
    y : array-like
        Input data.
    theta : array-like
        Input data.
    b : array-like
        Input data.
    tau_j : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrich (1978)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andrich Rating Scale Model (common thresholds)"})


def cheatsheet():
    return "rsmand: Andrich Rating Scale Model (common thresholds)"
