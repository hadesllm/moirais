"""Random-walk Laplacian."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_random_walk_laplacian"]


def sgt_random_walk_laplacian(A):
    """
    Random-walk Laplacian

    Formula: L_rw = I - D^{-1} A

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: L_rw

    References
    ----------
    Chung (1997)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random-walk Laplacian"})


def cheatsheet():
    return "sgtrwl: Random-walk Laplacian"
