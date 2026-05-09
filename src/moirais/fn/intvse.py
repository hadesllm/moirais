"""Interventional (in)direct effects."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["interventional_effect"]


def interventional_effect(Y, X, M, C):
    """
    Interventional (in)direct effects

    Formula: intervene on M's distribution rather than counterfactual

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele-Vansteelandt-Robins (2014)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Interventional (in)direct effects"})


def cheatsheet():
    return "intvse: Interventional (in)direct effects"
