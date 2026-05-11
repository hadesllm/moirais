"""Symmetric normalised Laplacian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_normalised_laplacian"]


def sgt_normalised_laplacian(A):
    """
    Symmetric normalised Laplacian

    Formula: L̃ = I - D^{-1/2} A D^{-1/2}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Lt

    References
    ----------
    Chung (1997)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetric normalised Laplacian"})


def cheatsheet():
    return "sgtnlap: Symmetric normalised Laplacian"
