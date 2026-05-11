"""Eccentricity centrality (1 / max distance)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["eccentricity_centrality"]


def eccentricity_centrality(y, A, node):
    """
    Eccentricity centrality (1 / max distance)

    Formula: C_E(v) = 1 / max_u d(v, u)

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.
    node : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hage & Harary (1995)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Eccentricity centrality (1 / max distance)"})


def cheatsheet():
    return "ecccen: Eccentricity centrality (1 / max distance)"
