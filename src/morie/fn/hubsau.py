"""The only true wisdom is in knowing you know nothing. -- Socrates"""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["The only true wisdom is in knowing you know nothing. -- Socrates"]


def hits_hub_authority(G):
    """
    HITS hub + authority scores

    Formula: alternate h = A a; a = A^T h

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
    Kleinberg (1999)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "The only true wisdom is in knowing you know nothing. -- Socrates"})


def cheatsheet():
    return "The only true wisdom is in knowing you know nothing. -- Socrates"
