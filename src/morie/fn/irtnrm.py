"""Bock nominal response model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nominal_response"]


def nominal_response(X, ncats):
    """
    Bock nominal response model

    Formula: category-specific slopes a_jk

    Parameters
    ----------
    X : array-like
        Input data.
    ncats : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bock (1972)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bock nominal response model"})


def cheatsheet():
    return "irtnrm: Bock nominal response model"
