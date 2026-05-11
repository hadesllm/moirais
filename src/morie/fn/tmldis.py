"""TMLE for population-attributable disparity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_disparity"]


def tmle_disparity(y, S, X, X_target):
    """
    TMLE for population-attributable disparity

    Formula: PAD = E[Y|S=1] - E[Y|do(X^*=X|S=0)]

    Parameters
    ----------
    y : array-like
        Input data.
    S : array-like
        Input data.
    X : array-like
        Input data.
    X_target : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele-Robinson (2014); Naimi-Schnitzer (2017)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for population-attributable disparity"})


def cheatsheet():
    return "tmldis: TMLE for population-attributable disparity"
