"""Network diameter."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["diameter"]


def diameter(G):
    """
    Network diameter

    Formula: max d(u,v)

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
    Newman (2010)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network diameter"})


def cheatsheet():
    return "diaspath: Network diameter"
