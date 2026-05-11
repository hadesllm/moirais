# morie.fn — function file (hadesllm/morie)
"""Cosine annealing learning rate schedule."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_cosine_annealing"]


def geron_cosine_annealing(t, T, eta_max, eta_min):
    """
    Cosine annealing learning rate schedule

    Formula: eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi*t/T))

    Parameters
    ----------
    t : array-like
        Input data.
    T : array-like
        Input data.
    eta_max : array-like
        Input data.
    eta_min : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eta

    References
    ----------
    Géron Ch 11
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cosine annealing learning rate schedule"})


def cheatsheet():
    return "hmlcos: Cosine annealing learning rate schedule"
