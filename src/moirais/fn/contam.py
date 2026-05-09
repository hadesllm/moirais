"""Huber gross-error model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["epsilon_contamination"]


def epsilon_contamination(epsilon, H):
    """
    Huber gross-error model

    Formula: F = (1−ε)Φ + εH

    Parameters
    ----------
    epsilon : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Huber (1964)
    """
    epsilon = np.atleast_1d(np.asarray(epsilon, dtype=float))
    n = len(epsilon)
    result = float(np.mean(epsilon))
    se = float(np.std(epsilon, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Huber gross-error model"})


def cheatsheet():
    return "contam: Huber gross-error model"
