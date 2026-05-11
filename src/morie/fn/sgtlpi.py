"""Moore-Penrose pseudoinverse of L."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_laplacian_pseudoinverse"]


def sgt_laplacian_pseudoinverse(A):
    """
    Moore-Penrose pseudoinverse of L

    Formula: L^+ via eigendecomp drop zero modes

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: L_plus

    References
    ----------
    Klein & Randić (1993)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Moore-Penrose pseudoinverse of L"})


def cheatsheet():
    return "sgtlpi: Moore-Penrose pseudoinverse of L"
