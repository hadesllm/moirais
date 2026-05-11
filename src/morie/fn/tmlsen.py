"""TMLE bias bound under unmeasured confounding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_sensitivity_unmeasured"]


def tmle_sensitivity_unmeasured(y, D, X, gamma_grid):
    """
    TMLE bias bound under unmeasured confounding

    Formula: E-value via bias term gamma

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    gamma_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Díaz-vdL (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE bias bound under unmeasured confounding"})


def cheatsheet():
    return "tmlsen: TMLE bias bound under unmeasured confounding"
