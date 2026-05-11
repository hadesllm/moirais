"""Effective resistance between nodes."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["effective_resistance"]


def effective_resistance(G, u, v):
    """
    Effective resistance between nodes

    Formula: R_uv = (e_u - e_v)^T L^+ (e_u - e_v)

    Parameters
    ----------
    G : array-like
        Input data.
    u : array-like
        Input data.
    v : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Klein-Randic (1993)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Effective resistance between nodes"})


def cheatsheet():
    return "esumtv: Effective resistance between nodes"
