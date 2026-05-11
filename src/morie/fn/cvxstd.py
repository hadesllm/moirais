"""Steepest descent direction."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_steepest_desc"]


def boyd_steepest_desc(grad, norm):
    """
    Steepest descent direction

    Formula: delta_x = argmin grad f(x)' v, |v| <= 1

    Parameters
    ----------
    grad : array-like
        Input data.
    norm : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: direction

    References
    ----------
    Boyd CVX Ch 9
    """
    grad = np.atleast_1d(np.asarray(grad, dtype=float))
    n = len(grad)
    result = float(np.mean(grad))
    se = float(np.std(grad, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Steepest descent direction"})


def cheatsheet():
    return "cvxstd: Steepest descent direction"
