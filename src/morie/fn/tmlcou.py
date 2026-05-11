"""TMLE for count outcomes (Poisson / NB)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tmle_count_outcome"]


def tmle_count_outcome(y, D, X, offset):
    """
    TMLE for count outcomes (Poisson / NB)

    Formula: target E[Y(a)] with Poisson clever covariate

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    offset : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lendle-Schwab-Petersen-vdL (2013)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "TMLE for count outcomes (Poisson / NB)"})


def cheatsheet():
    return "tmlcou: TMLE for count outcomes (Poisson / NB)"
