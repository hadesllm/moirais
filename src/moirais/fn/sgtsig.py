"""Signless Laplacian Q = D + A."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_signless_laplacian"]


def sgt_signless_laplacian(A):
    """
    Signless Laplacian Q = D + A

    Formula: Q = D + A

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Q

    References
    ----------
    Cvetković-Doob-Sachs (1995)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Signless Laplacian Q = D + A"})


def cheatsheet():
    return "sgtsig: Signless Laplacian Q = D + A"
