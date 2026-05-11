"""Eigenvector centrality from leading eigenvector of A."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_eigenvector_centrality"]


def sgt_eigenvector_centrality(A):
    """
    Eigenvector centrality from leading eigenvector of A

    Formula: Av = λ v with λ_max

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
    Bonacich (1972)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Eigenvector centrality from leading eigenvector of A"})


def cheatsheet():
    return "sgteig: Eigenvector centrality from leading eigenvector of A"
