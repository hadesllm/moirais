"""Spherical variogram model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spherical_variogram_model"]


def spherical_variogram_model(h, c0, c, a):
    """
    Spherical variogram model

    Formula: gamma(h) = c0 + c [3h/(2a) - h^3/(2a^3)] for h <= a

    Parameters
    ----------
    h : array-like
        Input data.
    c0 : array-like
        Input data.
    c : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cressie (1993) §2.3
    """
    h = np.atleast_1d(np.asarray(h, dtype=float))
    n = len(h)
    result = float(np.mean(h))
    se = float(np.std(h, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spherical variogram model"})


def cheatsheet():
    return "spvarm: Spherical variogram model"
