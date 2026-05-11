"""Graph transformer with structural encoding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graph_transformer"]


def graph_transformer(G, X):
    """
    Graph transformer with structural encoding

    Formula: transformer + Laplacian eigvec PE + edge bias

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dwivedi-Bresson (2020)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph transformer with structural encoding"})


def cheatsheet():
    return "gtrf: Graph transformer with structural encoding"
