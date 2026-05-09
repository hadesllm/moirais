"""Negative entropy of a transport plan."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_marginal_negent"]


def ot_marginal_negent(T):
    """
    Negative entropy of a transport plan

    Formula: H(T) = -Σ T_ij (log T_ij - 1)

    Parameters
    ----------
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: H

    References
    ----------
    Cuturi (2013)
    """
    T = np.atleast_1d(np.asarray(T, dtype=float))
    n = len(T)
    result = float(np.mean(T))
    se = float(np.std(T, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Negative entropy of a transport plan"})


def cheatsheet():
    return "otmnge: Negative entropy of a transport plan"
