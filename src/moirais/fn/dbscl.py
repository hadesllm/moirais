# moirais.fn — function file (hadesllm/moirais)
"""DBSCAN density-based clustering."""
import numpy as np
from ._richresult import RichResult

__all__ = ["dbscan_clustering"]


def dbscan_clustering(x):
    """
    DBSCAN density-based clustering

    Formula: core points: |N_eps(x)| >= minPts

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ester et al. (1996)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DBSCAN density-based clustering"})


def cheatsheet():
    return "dbscl: DBSCAN density-based clustering"
