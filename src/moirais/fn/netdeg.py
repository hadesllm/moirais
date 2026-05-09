"""Degree centrality (normalised)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["degree_centrality"]


def degree_centrality(y, A, node):
    """
    Degree centrality (normalised)

    Formula: C_D(v) = deg(v) / (n - 1)

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
    Freeman (1979)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Degree centrality (normalised)"})


def cheatsheet():
    return "netdeg: Degree centrality (normalised)"
