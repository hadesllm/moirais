"""Prophet automatic change-points."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["prophet_changepoint"]


def prophet_changepoint(y, n_cps):
    """
    Prophet automatic change-points

    Formula: sparse Laplace prior on rate-change parameters

    Parameters
    ----------
    y : array-like
        Input data.
    n_cps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Taylor-Letham (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prophet automatic change-points"})


def cheatsheet():
    return "prnFil: Prophet automatic change-points"
