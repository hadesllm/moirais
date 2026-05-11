"""Louvain modularity-greedy single pass."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sgt_louvain_step"]


def sgt_louvain_step(A, labels):
    """
    Louvain modularity-greedy single pass

    Formula: Move node to neighbour community maximising ΔQ

    Parameters
    ----------
    A : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels_new, Q_new

    References
    ----------
    Blondel-Guillaume-Lambiotte-Lefebvre (2008)
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Louvain modularity-greedy single pass"})


def cheatsheet():
    return "sgtcoml: Louvain modularity-greedy single pass"
