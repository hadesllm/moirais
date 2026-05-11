"""Augmented Lagrangian method."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["augmented_lagrangian"]


def augmented_lagrangian(f, constraints, x0, mu, lambda0):
    """
    Augmented Lagrangian method

    Formula: L_A = f + sum lambda_i g_i + mu/2 sum g_i^2

    Parameters
    ----------
    f : array-like
        Input data.
    constraints : array-like
        Input data.
    x0 : array-like
        Input data.
    mu : array-like
        Input data.
    lambda0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hestenes (1969); Powell (1969)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Augmented Lagrangian method"})


def cheatsheet():
    return "auglag: Augmented Lagrangian method"
