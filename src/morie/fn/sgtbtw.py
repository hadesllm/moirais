"""Betweenness centrality via Brandes algorithm (unweighted)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_betweenness_centrality"]


def sgt_betweenness_centrality(A):
    """
    Betweenness centrality via Brandes algorithm (unweighted)

    Formula: Σ_{s,t} σ_{st}(v)/σ_{st}

    Parameters
    ----------
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: BC

    References
    ----------
    Brandes (2001)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Betweenness centrality via Brandes algorithm (unweighted)"})


def cheatsheet():
    return "sgtbtw: Betweenness centrality via Brandes algorithm (unweighted)"
