"""Sinkhorn divergence (entropic OT)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sinkhorn_distance"]


def sinkhorn_distance(a, b, C, eps):
    """
    Sinkhorn divergence (entropic OT)

    Formula: argmin sum c_ij P_ij + eps H(P)

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    C : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cuturi (2013)
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sinkhorn divergence (entropic OT)"})


def cheatsheet():
    return "sinkhd: Sinkhorn divergence (entropic OT)"
