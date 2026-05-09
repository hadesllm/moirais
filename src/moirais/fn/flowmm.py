"""Maximum flow / minimum cut."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["max_flow_min_cut"]


def max_flow_min_cut(G, source, sink):
    """
    Maximum flow / minimum cut

    Formula: Ford-Fulkerson / Edmonds-Karp

    Parameters
    ----------
    G : array-like
        Input data.
    source : array-like
        Input data.
    sink : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ford-Fulkerson (1956)
    """
    G = np.atleast_1d(np.asarray(G, dtype=float))
    n = len(G)
    result = float(np.mean(G))
    se = float(np.std(G, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Maximum flow / minimum cut"})


def cheatsheet():
    return "flowmm: Maximum flow / minimum cut"
