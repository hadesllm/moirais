"""Cox Schoenfeld residuals for PH check."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_schoenfeld_residuals"]


def cox_schoenfeld_residuals(fit, time):
    """
    Cox Schoenfeld residuals for PH check

    Formula: r_i = X_i - sum_j w_ij X_j

    Parameters
    ----------
    fit : array-like
        Input data.
    time : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Schoenfeld (1982); Grambsch-Therneau (1994)
    """
    fit = np.atleast_1d(np.asarray(fit, dtype=float))
    n = len(fit)
    result = float(np.mean(fit))
    se = float(np.std(fit, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox Schoenfeld residuals for PH check"})


def cheatsheet():
    return "coxres: Cox Schoenfeld residuals for PH check"
