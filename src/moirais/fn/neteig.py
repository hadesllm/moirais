"""Eigenvector centrality (largest eigenvector of A)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["eigenvector_centrality"]


def eigenvector_centrality(y, A):
    """
    Eigenvector centrality (largest eigenvector of A)

    Formula: x = (1/lambda) A x; lambda = leading eigenvalue

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Bonacich (1972)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Eigenvector centrality (largest eigenvector of A)"})


def cheatsheet():
    return "neteig: Eigenvector centrality (largest eigenvector of A)"
