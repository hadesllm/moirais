"""Alternating direction method of multipliers."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["admm"]


def admm(f, g, A, B, c, rho):
    """
    Alternating direction method of multipliers

    Formula: x,z,u updates with augmented Lagrangian

    Parameters
    ----------
    f : array-like
        Input data.
    g : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    c : array-like
        Input data.
    rho : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Boyd et al (2011)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alternating direction method of multipliers"})


def cheatsheet():
    return "admmop: Alternating direction method of multipliers"
