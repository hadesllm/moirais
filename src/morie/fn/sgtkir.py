"""Kirchhoff index = total resistance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_kirchhoff_index"]


def sgt_kirchhoff_index(A):
    """
    Kirchhoff index = total resistance

    Formula: Kf = (1/2) Σ_{i,j} R_{ij} = n Σ 1/λ_k

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Kf

    References
    ----------
    Klein & Randić (1993)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kirchhoff index = total resistance"})


def cheatsheet():
    return "sgtkir: Kirchhoff index = total resistance"
