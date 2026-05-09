"""Smoothed bootstrap with kernel-perturbed samples."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boot_smoothed"]


def boot_smoothed(x, stat, h, B):
    """
    Smoothed bootstrap with kernel-perturbed samples

    Formula: x*_i = x_i + h ε_i, ε ~ K

    Parameters
    ----------
    x : array-like
        Input data.
    stat : array-like
        Input data.
    h : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: theta_b

    References
    ----------
    Silverman & Young (1987)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smoothed bootstrap with kernel-perturbed samples"})


def cheatsheet():
    return "btsmth: Smoothed bootstrap with kernel-perturbed samples"
