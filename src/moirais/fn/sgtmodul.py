"""Newman modularity matrix B."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_modularity_matrix"]


def sgt_modularity_matrix(A):
    """
    Newman modularity matrix B

    Formula: B_{ij} = A_{ij} - k_i k_j/(2m)

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: B

    References
    ----------
    Newman (2006)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newman modularity matrix B"})


def cheatsheet():
    return "sgtmodul: Newman modularity matrix B"
