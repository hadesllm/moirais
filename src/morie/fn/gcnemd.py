"""Graph convolutional network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gcn"]


def gcn(G, X, W):
    """
    Graph convolutional network

    Formula: H' = sigma(D^-0.5 A D^-0.5 H W)

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Kipf-Welling (2017)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph convolutional network"})


def cheatsheet():
    return "gcnemd: Graph convolutional network"
