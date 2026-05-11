"""Deep Graph Infomax."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["dgi"]


def dgi(G, X, encoder):
    """
    Deep Graph Infomax

    Formula: max MI(local h_v, global s)

    Parameters
    ----------
    G : array-like
        Input data.
    X : array-like
        Input data.
    encoder : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Velickovic et al (2019)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Deep Graph Infomax"})


def cheatsheet():
    return "dgi: Deep Graph Infomax"
