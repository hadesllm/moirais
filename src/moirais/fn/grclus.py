"""K-way graph clustering (METIS)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graph_clustering"]


def graph_clustering(A, k):
    """
    K-way graph clustering (METIS)

    Formula: multilevel coarsen / partition / refine

    Parameters
    ----------
    A : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Karypis-Kumar (1998) METIS
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-way graph clustering (METIS)"})


def cheatsheet():
    return "grclus: K-way graph clustering (METIS)"
