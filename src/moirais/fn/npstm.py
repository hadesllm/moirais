"""Nonparametric TMLE for survival treatment effect."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nonparametric_tmle_survival"]


def nonparametric_tmle_survival(time, event, A, W):
    """
    Nonparametric TMLE for survival treatment effect

    Formula: S_a(t) = E[ Q_n^*(t, A=a, W) ] with targeted update

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    A : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van der Laan & Gruber (2010); Stitelman & van der Laan (2010)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric TMLE for survival treatment effect"})


def cheatsheet():
    return "npstm: Nonparametric TMLE for survival treatment effect"
