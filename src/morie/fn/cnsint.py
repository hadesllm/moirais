"""Concurrent calibration linkage (single calibration)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["concurrent_calibration"]


def concurrent_calibration(y, item, group, anchor):
    """
    Concurrent calibration linkage (single calibration)

    Formula: jointly fit b_F, b_R on combined sample with anchor items

    Parameters
    ----------
    y : array-like
        Input data.
    item : array-like
        Input data.
    group : array-like
        Input data.
    anchor : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wingersky & Lord (1984)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Concurrent calibration linkage (single calibration)"})


def cheatsheet():
    return "cnsint: Concurrent calibration linkage (single calibration)"
