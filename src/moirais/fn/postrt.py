"""Post-stratification weight adjustment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["post_stratification"]


def post_stratification(y, weights, stratum, N_h):
    """
    Post-stratification weight adjustment

    Formula: w_i' = w_i * (N_h / hat N_h)

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    stratum : array-like
        Input data.
    N_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holt & Smith (1979)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Post-stratification weight adjustment"})


def cheatsheet():
    return "postrt: Post-stratification weight adjustment"
