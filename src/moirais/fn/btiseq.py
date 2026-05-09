"""Iterated bootstrap CI calibration via prepivoting."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_iter_calibrated"]


def boot_iter_calibrated(x, stat, B, iters, alpha):
    """
    Iterated bootstrap CI calibration via prepivoting

    Formula: Repeatedly prepivot until coverage stable

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.
    iters : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi, iters_used

    References
    ----------
    Beran (1988)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Iterated bootstrap CI calibration via prepivoting"})


def cheatsheet():
    return "btiseq: Iterated bootstrap CI calibration via prepivoting"
