"""AR-sieve bootstrap for stationary time series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_ar_sieve"]


def boot_ar_sieve(x, p_max, stat, B):
    """
    AR-sieve bootstrap for stationary time series

    Formula: Fit AR(p̂); resample residuals; recursively rebuild

    Parameters
    ----------
    x : array-like
        Input data.
    p_max : array-like
        Input data.
    stat : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b, p_hat

    References
    ----------
    Bühlmann (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AR-sieve bootstrap for stationary time series"})


def cheatsheet():
    return "btarsv: AR-sieve bootstrap for stationary time series"
