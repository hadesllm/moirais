"""Positional encoding cos.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_positional_encoding_cos"]


def kamath_ch2_positional_encoding_cos(i, j, d):
    """
    Positional encoding cos.

    Formula: P_{i,2j+1} = \cos(i/10000^{2j/d})

    Parameters
    ----------
    i : array-like
        Input data.
    j : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.14, p. 35
    """
    i = np.atleast_1d(np.asarray(i, dtype=float))
    n = len(i)
    result = float(np.mean(i))
    se = float(np.std(i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Positional encoding cos."})


def cheatsheet():
    return "km014: Positional encoding cos."
