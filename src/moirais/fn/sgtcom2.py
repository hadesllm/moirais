"""Estrada communicability matrix exp(A)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_communicability_matrix"]


def sgt_communicability_matrix(A):
    """
    Estrada communicability matrix exp(A)

    Formula: C = exp(A)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: C

    References
    ----------
    Estrada-Hatano (2008)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Estrada communicability matrix exp(A)"})


def cheatsheet():
    return "sgtcom2: Estrada communicability matrix exp(A)"
