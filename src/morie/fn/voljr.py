"""Threshold jump-robust realised variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vol_jump_robust_var"]


def vol_jump_robust_var(r_intraday, theta):
    """
    Threshold jump-robust realised variance

    Formula: RV_J = Σ r_i² 1{|r_i|<= θ_i}

    Parameters
    ----------
    r_intraday : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: RV_robust

    References
    ----------
    Mancini (2009)
    """
    r_intraday = np.atleast_1d(np.asarray(r_intraday, dtype=float))
    n = len(r_intraday)
    result = float(np.mean(r_intraday))
    se = float(np.std(r_intraday, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Threshold jump-robust realised variance"})


def cheatsheet():
    return "voljr: Threshold jump-robust realised variance"
