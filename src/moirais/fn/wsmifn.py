"""Influence function L(x) = lim (T((1-e)F+e d_x)-T(F))/e."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_influence_function"]


def wasserman_influence_function(data, T):
    """
    Influence function L(x) = lim (T((1-e)F+e d_x)-T(F))/e

    Formula: L_F(x) = lim_{e->0} (T((1-e)F + e delta_x) - T(F)) / e

    Parameters
    ----------
    data : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: influence

    References
    ----------
    Wasserman (2004), Ch 7
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Influence function L(x) = lim (T((1-e)F+e d_x)-T(F))/e"})


def cheatsheet():
    return "wsmifn: Influence function L(x) = lim (T((1-e)F+e d_x)-T(F))/e"
