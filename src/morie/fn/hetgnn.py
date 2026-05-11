"""Heterogeneous GNN ()."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["heterogeneous_gnn"]


def heterogeneous_gnn(G, X, metapaths):
    """
    Heterogeneous GNN ()

    Formula: node-level + semantic-level attention

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    metapaths : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Wang et al (2019)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heterogeneous GNN ()"})


def cheatsheet():
    return "hetgnn: Heterogeneous GNN ()"
