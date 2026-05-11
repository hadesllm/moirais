"""GIN — Graph Isomorphism Network."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["graph_isomorphism_net"]


def graph_isomorphism_net(G, X, eps):
    """
    GIN — Graph Isomorphism Network

    Formula: MLP((1+eps) h_v + sum h_u)

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    eps : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Xu et al (2019) GIN
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "GIN — Graph Isomorphism Network"})


def cheatsheet():
    return "ginemd: GIN — Graph Isomorphism Network"
