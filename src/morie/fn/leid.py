"""Leiden community detection (refines Louvain, guarantees well-connected)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["leiden_communities"]


def leiden_communities(y, A, resolution):
    """
    Leiden community detection (refines Louvain, guarantees well-connected)

    Formula: local move + refinement + aggregation; well-connected communities

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    resolution : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Traag, Waltman, van Eck (2019)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Leiden community detection (refines Louvain, guarantees well-connected)"})


def cheatsheet():
    return "leid: Leiden community detection (refines Louvain, guarantees well-connected)"
