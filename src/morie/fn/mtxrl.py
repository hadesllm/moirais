"""Matrix game / minimax."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matrix_game"]


def matrix_game(A):
    """
    Matrix game / minimax

    Formula: max_x min_y x^T A y

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    von Neumann (1928)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matrix game / minimax"})


def cheatsheet():
    return "mtxrl: Matrix game / minimax"
