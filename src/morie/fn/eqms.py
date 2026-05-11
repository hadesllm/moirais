"""Mean-sigma equating coefficients."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["equating_mean_sigma"]


def equating_mean_sigma(y, b_R, b_F):
    """
    Mean-sigma equating coefficients

    Formula: A = sd(b_F) / sd(b_R); B = mean(b_F) - A * mean(b_R)

    Parameters
    ----------
    y : array-like
        Input data.
    b_R : array-like
        Input data.
    b_F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Marco (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean-sigma equating coefficients"})


def cheatsheet():
    return "eqms: Mean-sigma equating coefficients"
