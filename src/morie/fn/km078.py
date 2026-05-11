"""Alignment function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_alignment_function"]


def kamath_ch6_alignment_function(a, b, y):
    """
    Alignment function.

    Formula: f: (a,b) \to y

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.2, p. 220
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alignment function."})


def cheatsheet():
    return "km078: Alignment function."
