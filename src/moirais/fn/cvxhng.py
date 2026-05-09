"""Hinge loss."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_hinge_loss"]


def boyd_hinge_loss(u):
    """
    Hinge loss

    Formula: phi(u) = max(0, 1 - u)

    Parameters
    ----------
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Boyd CVX Ch 6
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    n = len(u)
    result = float(np.mean(u))
    se = float(np.std(u, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hinge loss"})


def cheatsheet():
    return "cvxhng: Hinge loss"
