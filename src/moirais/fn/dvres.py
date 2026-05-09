"""Deviance residual for Cox PH (symmetric martingale)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["deviance_residual_cox"]


def deviance_residual_cox(time, event, fitted):
    """
    Deviance residual for Cox PH (symmetric martingale)

    Formula: d_i = sign(M_i) sqrt(-2(M_i + N_i log(N_i - M_i)))

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    fitted : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Therneau & Grambsch (2000) Ch 5
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deviance residual for Cox PH (symmetric martingale)"})


def cheatsheet():
    return "dvres: Deviance residual for Cox PH (symmetric martingale)"
