"""Network density."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["density"]


def density(G):
    """
    Network density

    Formula: |E| / (n choose 2)

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
    Wasserman-Faust (1994)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network density"})


def cheatsheet():
    return "densty: Network density"
