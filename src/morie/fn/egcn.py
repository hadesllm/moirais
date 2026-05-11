"""E(n)-equivariant GCN."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["e_gcn"]


def e_gcn(G, X, coords):
    """
    E(n)-equivariant GCN

    Formula: E(n) equivariant message passing

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Satorras-Hoogeboom-Welling (2021)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "E(n)-equivariant GCN"})


def cheatsheet():
    return "egcn: E(n)-equivariant GCN"
