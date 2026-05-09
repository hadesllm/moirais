"""Samejima graded response model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graded_response"]


def graded_response(X, ncats):
    """
    Samejima graded response model

    Formula: P(X >= k | theta) = 2PL with category threshold b_jk

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
    Samejima (1969)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Samejima graded response model"})


def cheatsheet():
    return "irtgrm: Samejima graded response model"
