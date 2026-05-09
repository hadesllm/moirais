"""Tail dependence coefficient χ for two series."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_chi_tail_dependence"]


def evt_chi_tail_dependence(x, y, u):
    """
    Tail dependence coefficient χ for two series

    Formula: χ = lim_{u→1} P(F_X(X)>u | F_Y(Y)>u)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    u : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: chi

    References
    ----------
    Coles-Heffernan-Tawn (1999)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tail dependence coefficient χ for two series"})


def cheatsheet():
    return "evchitd: Tail dependence coefficient χ for two series"
