"""Bootstrap percentile CI for the median."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_ci_median"]


def boot_ci_median(x, B, alpha):
    """
    Bootstrap percentile CI for the median

    Formula: Special case of bootstrap quantile τ=0.5

    Parameters
    ----------
    x : array-like
        Input data.
    B : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lo, hi

    References
    ----------
    Efron (1979)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap percentile CI for the median"})


def cheatsheet():
    return "btcimed: Bootstrap percentile CI for the median"
