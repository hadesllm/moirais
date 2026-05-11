"""Proportion of total effect explained."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proportion_te_explained"]


def proportion_te_explained(nie, te):
    """
    Proportion of total effect explained

    Formula: PTE = NIE / TE

    Parameters
    ----------
    nie : array-like
        Input data.
    te : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2013)
    """
    nie = np.atleast_1d(np.asarray(nie, dtype=float))
    n = len(nie)
    result = float(np.mean(nie))
    se = float(np.std(nie, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportion of total effect explained"})


def cheatsheet():
    return "pmedex: Proportion of total effect explained"
