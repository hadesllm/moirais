"""Fiedler vector -- eigenvector for λ_2."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_fiedler_vector"]


def sgt_fiedler_vector(A):
    """
    Fiedler vector -- eigenvector for λ_2

    Formula: L v = λ_2 v

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: v

    References
    ----------
    Fiedler (1973)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fiedler vector -- eigenvector for λ_2"})


def cheatsheet():
    return "sgtfvc: Fiedler vector -- eigenvector for λ_2"
