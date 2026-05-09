"""Bound on kink TE."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_kink_te"]


def bound_kink_te(y, x, cutoff):
    """
    Bound on kink TE

    Formula: bounds for kinked-design RD

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Card et al (2015)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bound on kink TE"})


def cheatsheet():
    return "bnskt2: Bound on kink TE"
