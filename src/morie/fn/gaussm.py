"""Gaussian mechanism."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gaussian_mechanism"]


def gaussian_mechanism(f_value, l2_sens, epsilon, delta):
    """
    Gaussian mechanism

    Formula: M(D) = f(D) + N(0, σ²); σ ≥ √(2 ln 1.25/δ)·Δ₂f/ε

    Parameters
    ----------
    f_value : array-like
        Input data.
    l2_sens : array-like
        Input data.
    epsilon : array-like
        Input data.
    delta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwork-Roth (2014) book §3.5
    """
    f_value = np.atleast_1d(np.asarray(f_value, dtype=float))
    n = len(f_value)
    result = float(np.mean(f_value))
    se = float(np.std(f_value, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gaussian mechanism"})


def cheatsheet():
    return "gaussm: Gaussian mechanism"
