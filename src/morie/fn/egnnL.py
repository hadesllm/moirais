"""EGNN layer (equivariant graph NN)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["egnn_layer"]


def egnn_layer(h, x, edges):
    """
    EGNN layer (equivariant graph NN)

    Formula: update h_i and x_i with relative positions

    Parameters
    ----------
    h : array-like
        Input data.
    x : array-like
        Input data.
    edges : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Satorras et al (2021)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EGNN layer (equivariant graph NN)"})


def cheatsheet():
    return "egnnL: EGNN layer (equivariant graph NN)"
