"""Network transitivity."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["transitivity"]


def transitivity(G):
    """
    Network transitivity

    Formula: 3 * triangles / triads

    Parameters
    ----------
    G : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Newman (2003)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network transitivity"})


def cheatsheet():
    return "trnscl: Network transitivity"
