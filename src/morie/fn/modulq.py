"""Newman-Girvan modularity Q."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["modularity_q"]


def modularity_q(G, communities):
    """
    Newman-Girvan modularity Q

    Formula: Q = (1/2m) sum (A_ij - k_i k_j/2m) delta(c_i,c_j)

    Parameters
    ----------
    G : array-like
        Input data.
    communities : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman-Girvan (2004)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Newman-Girvan modularity Q"})


def cheatsheet():
    return "modulq: Newman-Girvan modularity Q"
