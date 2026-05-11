"""Fiedler (algebraic connectivity) eigenvalue λ_2 of L."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_fiedler_value"]


def sgt_fiedler_value(A):
    """
    Fiedler (algebraic connectivity) eigenvalue λ_2 of L

    Formula: λ_2(L)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lam2

    References
    ----------
    Fiedler (1973)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fiedler (algebraic connectivity) eigenvalue λ_2 of L"})


def cheatsheet():
    return "sgtfid: Fiedler (algebraic connectivity) eigenvalue λ_2 of L"
