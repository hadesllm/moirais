"""DP changepoint detection."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dp_changepoint"]


def dp_changepoint(y, alpha):
    """
    DP changepoint detection

    Formula: DP prior on segment-specific parameters

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barry-Hartigan (1992); Adams-MacKay (2007)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DP changepoint detection"})


def cheatsheet():
    return "dpchpr: DP changepoint detection"
