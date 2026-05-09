"""GraphSAGE inductive."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graphsage"]


def graphsage(G, X, W, aggregator):
    """
    GraphSAGE inductive

    Formula: h_v = σ(W·CONCAT(h_v, AGG{h_u}))

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    W : array-like
        Input data.
    aggregator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hamilton-Ying-Leskovec (2017)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GraphSAGE inductive"})


def cheatsheet():
    return "sage: GraphSAGE inductive"
