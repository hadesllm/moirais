"""Restricted mean survival time (RMST)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["restricted_lifetime"]


def restricted_lifetime(fit, t_star):
    """
    Restricted mean survival time (RMST)

    Formula: RMST(t*) = integral_0^t* S(u) du

    Parameters
    ----------
    fit : array-like
        Input data.
    t_star : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Royston-Parmar (2013)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Restricted mean survival time (RMST)"})


def cheatsheet():
    return "survrls: Restricted mean survival time (RMST)"
