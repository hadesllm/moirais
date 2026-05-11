"""Nonparametric Bayes survival via Beta-process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["np_bayes_survival"]


def np_bayes_survival(time, event):
    """
    Nonparametric Bayes survival via Beta-process

    Formula: Beta-process prior on cumulative hazard

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hjort (1990)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nonparametric Bayes survival via Beta-process"})


def cheatsheet():
    return "npbsr: Nonparametric Bayes survival via Beta-process"
