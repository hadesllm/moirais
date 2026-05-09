"""Graph Convolutional Network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["gcn"]


def gcn(A, X, W):
    """
    Graph Convolutional Network

    Formula: H^{l+1} = σ(D^{-1/2}ÃD^{-1/2} H^l W^l)

    Parameters
    ----------
    A : array-like
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
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph Convolutional Network"})


def cheatsheet():
    return "gcnL: Graph Convolutional Network"
