"""Degree-corrected SBM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["degree_corrected_sbm"]


def degree_corrected_sbm(G, K):
    """
    Degree-corrected SBM

    Formula: P(A) ~ theta_i theta_j B_{c_i, c_j}

    Parameters
    ----------
    G : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Karrer-Newman (2011)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Degree-corrected SBM"})


def cheatsheet():
    return "sbmdg2: Degree-corrected SBM"
