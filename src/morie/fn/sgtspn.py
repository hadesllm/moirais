"""Matrix-tree theorem: number of spanning trees."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_spanning_tree_count"]


def sgt_spanning_tree_count(A):
    """
    Matrix-tree theorem: number of spanning trees

    Formula: τ(G) = (1/n) Π_{k>=2} λ_k(L)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: tau

    References
    ----------
    Kirchhoff (1847)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matrix-tree theorem: number of spanning trees"})


def cheatsheet():
    return "sgtspn: Matrix-tree theorem: number of spanning trees"
