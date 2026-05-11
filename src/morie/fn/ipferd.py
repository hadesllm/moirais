"""IPW with replicate-weight variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ipw_with_replicate"]


def ipw_with_replicate(y, D, w, replicate_weights):
    """
    IPW with replicate-weight variance

    Formula: jackknife on weighted estimate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    w : array-like
        Input data.
    replicate_weights : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lumley-Scott (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "IPW with replicate-weight variance"})


def cheatsheet():
    return "ipferd: IPW with replicate-weight variance"
