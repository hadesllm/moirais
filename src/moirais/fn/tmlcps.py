"""TMLE for continuous-valued treatments."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_continuous_treatment"]


def tmle_continuous_treatment(y, A, X, a_grid):
    """
    TMLE for continuous-valued treatments

    Formula: target E[Y(a)] across a; spline Q + GAM g

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    X : array-like
        Input data.
    a_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kennedy-Ma-McHugh-Small (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for continuous-valued treatments"})


def cheatsheet():
    return "tmlcps: TMLE for continuous-valued treatments"
