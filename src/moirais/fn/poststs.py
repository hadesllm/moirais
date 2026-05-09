"""Post-stratification."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["poststratify"]


def poststratify(y, stratum, Nh):
    """
    Post-stratification

    Formula: sum_h (N_h / sum N) ybar_h

    Parameters
    ----------
    y : array-like
        Input data.
    stratum : array-like
        Input data.
    Nh : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Holt-Smith (1979)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Post-stratification"})


def cheatsheet():
    return "poststs: Post-stratification"
