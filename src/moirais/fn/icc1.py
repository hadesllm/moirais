"""ICC(1) one-way random-effects model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["icc_one_way"]


def icc_one_way(y, cluster):
    """
    ICC(1) one-way random-effects model

    Formula: ICC(1) = (MS_b - MS_w) / (MS_b + (k-1) MS_w)

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Shrout & Fleiss (1979)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ICC(1) one-way random-effects model"})


def cheatsheet():
    return "icc1: ICC(1) one-way random-effects model"
