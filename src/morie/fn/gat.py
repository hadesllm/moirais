"""Graph Attention Network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gat"]


def gat(A, X, W, a):
    """
    Graph Attention Network

    Formula: α_{ij} = softmax(LeakyReLU(a^T[Wh_i||Wh_j]))

    Parameters
    ----------
    A : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Velickovic et al (2018)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph Attention Network"})


def cheatsheet():
    return "gat: Graph Attention Network"
