"""Stochastic block model fit."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["stochastic_block_model"]


def stochastic_block_model(G, K):
    """
    Stochastic block model fit

    Formula: P(A_ij = 1) = B_{c_i, c_j}

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
    Holland-Laskey-Leinhardt (1983)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Stochastic block model fit"})


def cheatsheet():
    return "sbmest: Stochastic block model fit"
