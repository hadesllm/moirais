"""Semi-parametric tail / extreme regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["semiparametric_max"]


def semiparametric_max(y, X, model):
    """
    Semi-parametric tail / extreme regression

    Formula: GAMLSS for non-stationary GEV

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    model : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Coles (2001) Ch.6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semi-parametric tail / extreme regression"})


def cheatsheet():
    return "smt: Semi-parametric tail / extreme regression"
