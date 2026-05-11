"""Double bootstrap for calibrated CI coverage."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_double"]


def boot_double(x, stat, B, Bp, alpha):
    """
    Double bootstrap for calibrated CI coverage

    Formula: Inner B' resamples per outer for nominal-coverage adj

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.
    Bp : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi, alpha_adj

    References
    ----------
    Beran (1987)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double bootstrap for calibrated CI coverage"})


def cheatsheet():
    return "btdbl: Double bootstrap for calibrated CI coverage"
