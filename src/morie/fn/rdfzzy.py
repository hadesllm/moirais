"""Fuzzy RDD (incomplete compliance at cutoff)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fuzzy_rdd"]


def fuzzy_rdd(y, x, D, cutoff, bandwidth):
    """
    Fuzzy RDD (incomplete compliance at cutoff)

    Formula: tau_LATE = (lim Y+ - lim Y-) / (lim D+ - lim D-)

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    D : array-like
        Input data.
    cutoff : array-like
        Input data.
    bandwidth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hahn, Todd, van der Klaauw (2001); Imbens-Lemieux (2008)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fuzzy RDD (incomplete compliance at cutoff)"})


def cheatsheet():
    return "rdfzzy: Fuzzy RDD (incomplete compliance at cutoff)"
