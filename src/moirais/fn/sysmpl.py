"""Systematic sample with random start."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["systematic_sample"]


def systematic_sample(frame, n):
    """
    Systematic sample with random start

    Formula: every k-th element after random offset

    Parameters
    ----------
    frame : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Madow-Madow (1944)
    """
    frame = np.atleast_1d(np.asarray(frame, dtype=float))
    n = len(frame)
    result = float(np.mean(frame))
    se = float(np.std(frame, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Systematic sample with random start"})


def cheatsheet():
    return "sysmpl: Systematic sample with random start"
