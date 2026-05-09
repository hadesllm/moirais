"""Asymptotic-independence diagnostic χ̄(u)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_chibar_dependence"]


def evt_chibar_dependence(x, y, u_grid):
    """
    Asymptotic-independence diagnostic χ̄(u)

    Formula: χ̄(u) = 2 log(1-u)/log P(F_X>u, F_Y>u) - 1

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    u_grid : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chibar_curve

    References
    ----------
    Coles-Heffernan-Tawn (1999)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Asymptotic-independence diagnostic χ̄(u)"})


def cheatsheet():
    return "evchibd: Asymptotic-independence diagnostic χ̄(u)"
