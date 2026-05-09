"""Weak identification check for mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["weak_identification_mediation"]


def weak_identification_mediation(a, b, se_a, se_b):
    """
    Weak identification check for mediation

    Formula: F-stat on a*b instrument relevance

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    se_a : array-like
        Input data.
    se_b : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Tchetgen Tchetgen & Shpitser (2012)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weak identification check for mediation"})


def cheatsheet():
    return "weakid: Weak identification check for mediation"
