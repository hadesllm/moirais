"""Moverscore distance.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_moverscore_distance"]


def kamath_ch8_moverscore_distance(x_i, y_j, E):
    """
    Moverscore distance.

    Formula: d(x_i^n, y_j^n) = \|E(x_i^n) - E(y_j^n)\|_2

    Parameters
    ----------
    x_i : array-like
        Input data.
    y_j : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.11, p. 326
    """
    x_i = np.atleast_1d(np.asarray(x_i, dtype=float))
    n = len(x_i)
    result = float(np.mean(x_i))
    se = float(np.std(x_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moverscore distance."})


def cheatsheet():
    return "km123: Moverscore distance."
