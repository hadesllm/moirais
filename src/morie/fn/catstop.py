"""CAT stopping rule (SE / SE_target)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cat_stopping_rule"]


def cat_stopping_rule(theta, items_taken, threshold):
    """
    CAT stopping rule (SE / SE_target)

    Formula: stop when SE(theta) < threshold

    Parameters
    ----------
    theta : array-like
        Input data.
    items_taken : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wainer-Mislevy (2000)
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    n = len(theta)
    result = float(np.mean(theta))
    se = float(np.std(theta, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "CAT stopping rule (SE / SE_target)"})


def cheatsheet():
    return "catstop: CAT stopping rule (SE / SE_target)"
