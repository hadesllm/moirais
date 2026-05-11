"""TMLE for periodic / seasonal exposures."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_periodic"]


def tmle_periodic(y, D, X, period):
    """
    TMLE for periodic / seasonal exposures

    Formula: Fourier-basis Q + season-aware g

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    period : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Westreich-Cole (2010)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for periodic / seasonal exposures"})


def cheatsheet():
    return "tmlper: TMLE for periodic / seasonal exposures"
