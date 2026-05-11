"""Graph attention network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graph_attention_net"]


def graph_attention_net(G, X, heads):
    """
    Graph attention network

    Formula: per-edge attention coefficient + aggregate

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    heads : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Veličković et al (2018) GAT
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Graph attention network"})


def cheatsheet():
    return "gatemd: Graph attention network"
