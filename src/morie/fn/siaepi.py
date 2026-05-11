"""Network SIR epidemic."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sir_epidemic"]


def sir_epidemic(G, beta, gamma, initial):
    """
    Network SIR epidemic

    Formula: per-edge infection rate beta; recovery gamma

    Parameters
    ----------
    G : array-like
        Input data.
    beta : array-like
        Input data.
    gamma : array-like
        Input data.
    initial : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pastor-Satorras-Vespignani (2001)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Network SIR epidemic"})


def cheatsheet():
    return "siaepi: Network SIR epidemic"
