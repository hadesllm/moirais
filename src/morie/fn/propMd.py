"""Proportion mediated NIE / TE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["proportion_mediated"]


def proportion_mediated(NIE, NDE):
    """
    Proportion mediated NIE / TE

    Formula: PM = NIE / (NIE + NDE)

    Parameters
    ----------
    NIE : array-like
        Input data.
    NDE : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    VanderWeele (2015)
    """
    NIE = np.atleast_1d(np.asarray(NIE, dtype=float))
    n = len(NIE)
    result = float(np.mean(NIE))
    se = float(np.std(NIE, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Proportion mediated NIE / TE"})


def cheatsheet():
    return "propMd: Proportion mediated NIE / TE"
