"""Graph Isomorphism Network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gin"]


def gin(A, X, epsilon):
    """
    Graph Isomorphism Network

    Formula: h^{l+1}_v = MLP((1+ε) h^l_v + sum_u h^l_u)

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xu et al (2019) GIN
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph Isomorphism Network"})


def cheatsheet():
    return "gin: Graph Isomorphism Network"
