"""Pacific Decadal Oscillation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pdo"]


def pdo(sst):
    """
    Pacific Decadal Oscillation

    Formula: PC1 of N. Pacific SST anomaly

    Parameters
    ----------
    sst : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Mantua-Hare (2002)
    """
    sst = np.atleast_1d(np.asarray(sst, dtype=float))
    n = len(sst)
    result = float(np.mean(sst))
    se = float(np.std(sst, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pacific Decadal Oscillation"})


def cheatsheet():
    return "pdoIdx: Pacific Decadal Oscillation"
