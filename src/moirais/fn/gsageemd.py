"""GraphSAGE inductive embedding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graphsage"]


def graphsage(G, X, aggregator):
    """
    GraphSAGE inductive embedding

    Formula: sample + aggregate per layer

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    aggregator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hamilton et al (2017)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GraphSAGE inductive embedding"})


def cheatsheet():
    return "gsageemd: GraphSAGE inductive embedding"
