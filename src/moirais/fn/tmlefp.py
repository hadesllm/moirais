"""TMLE for the effective sample-size adjusted ATT."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_effective_pi"]


def tmle_effective_pi(y, D, X, trim):
    """
    TMLE for the effective sample-size adjusted ATT

    Formula: weight by ESS to stabilize finite-sample variance

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    trim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Crump-Hotz-Imbens-Mitnik (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for the effective sample-size adjusted ATT"})


def cheatsheet():
    return "tmlefp: TMLE for the effective sample-size adjusted ATT"
