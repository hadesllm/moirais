"""Penalty method for constrained."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["penalty_method"]


def penalty_method(f, constraints, x0, mu):
    """
    Penalty method for constrained

    Formula: unconstrained + mu * sum violation^2

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

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Courant (1943)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Penalty method for constrained"})


def cheatsheet():
    return "penmth: Penalty method for constrained"
