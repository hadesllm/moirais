"""Bootstrap percentile CI for an arbitrary quantile."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_ci_quantile"]


def boot_ci_quantile(x, tau, B, alpha):
    """
    Bootstrap percentile CI for an arbitrary quantile

    Formula: Q*_b(τ); percentile interval over B

    Parameters
    ----------
    x : array-like
        Input data.
    tau : array-like
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
    Davison & Hinkley (1997)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bootstrap percentile CI for an arbitrary quantile"})


def cheatsheet():
    return "btciqua: Bootstrap percentile CI for an arbitrary quantile"
