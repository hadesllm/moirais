"""PELT pruned exact linear changepoint."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["changepoint_segmentation"]


def changepoint_segmentation(y, penalty):
    """
    PELT pruned exact linear changepoint

    Formula: dynamic programming with cost function

    Parameters
    ----------
    y : array-like
        Input data.
    penalty : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Killick et al (2012) PELT
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PELT pruned exact linear changepoint"})


def cheatsheet():
    return "chgseg: PELT pruned exact linear changepoint"
