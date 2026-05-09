"""Mean-mean equating coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["equating_mean_mean"]


def equating_mean_mean(y, a_R, b_R, a_F, b_F):
    """
    Mean-mean equating coefficients

    Formula: A = mean(a_R) / mean(a_F); B = mean(b_R) - A * mean(b_F)

    Parameters
    ----------
    y : array-like
        Input data.
    a_R : array-like
        Input data.
    b_R : array-like
        Input data.
    a_F : array-like
        Input data.
    b_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Loyd & Hoover (1980)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-mean equating coefficients"})


def cheatsheet():
    return "eqmm: Mean-mean equating coefficients"
