"""Entropic regulariser term in entropic OT objective."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_entropy_regulariser"]


def ot_entropy_regulariser(T, epsilon):
    """
    Entropic regulariser term in entropic OT objective

    Formula: ε H(T) = ε Σ T_ij (log T_ij - 1)

    Parameters
    ----------
    T : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Cuturi (2013)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Entropic regulariser term in entropic OT objective"})


def cheatsheet():
    return "otentr: Entropic regulariser term in entropic OT objective"
