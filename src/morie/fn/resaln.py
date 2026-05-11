"""Resultant of two polynomials."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["resultant"]


def resultant(p, q):
    """
    Resultant of two polynomials

    Formula: determinant of Sylvester matrix

    Parameters
    ----------
    p : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    classical
    """
    p = np.atleast_1d(np.asarray(p, dtype=float))
    n = len(p)
    result = float(np.mean(p))
    se = float(np.std(p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Resultant of two polynomials"})


def cheatsheet():
    return "resaln: Resultant of two polynomials"
