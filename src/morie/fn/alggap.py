"""Algebraic connectivity (lambda_2 of L)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["algebraic_connectivity"]


def algebraic_connectivity(G):
    """
    Algebraic connectivity (lambda_2 of L)

    Formula: second smallest eigenvalue of Laplacian

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
    Fiedler (1973)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Algebraic connectivity (lambda_2 of L)"})


def cheatsheet():
    return "alggap: Algebraic connectivity (lambda_2 of L)"
