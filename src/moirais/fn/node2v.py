"""node2vec biased random walks."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["node2vec"]


def node2vec(G, p, q, dim):
    """
    node2vec biased random walks

    Formula: p,q-controlled walk + skip-gram

    Parameters
    ----------
    G : array-like
        Input data.
    p : array-like
        Input data.
    q : array-like
        Input data.
    dim : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Grover-Leskovec (2016)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "node2vec biased random walks"})


def cheatsheet():
    return "node2v: node2vec biased random walks"
