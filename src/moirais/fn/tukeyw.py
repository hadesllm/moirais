"""Tukey biweight (bisquare) weight function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["tukey_biweight"]


def tukey_biweight(y, c):
    """
    Tukey biweight (bisquare) weight function

    Formula: w(r) = (1 - (r/c)^2)^2 if |r|<=c else 0

    Parameters
    ----------
    y : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Beaton & Tukey (1974)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tukey biweight (bisquare) weight function"})


def cheatsheet():
    return "tukeyw: Tukey biweight (bisquare) weight function"
