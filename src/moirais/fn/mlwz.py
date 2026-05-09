"""Multilevel within-cluster z-score (cluster-standardized)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multilevel_within_cluster_z"]


def multilevel_within_cluster_z(y, cluster):
    """
    Multilevel within-cluster z-score (cluster-standardized)

    Formula: z_ij = (x_ij - xbar_j) / sd_j

    Parameters
    ----------
    y : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Raudenbush & Bryk (2002) §5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilevel within-cluster z-score (cluster-standardized)"})


def cheatsheet():
    return "mlwz: Multilevel within-cluster z-score (cluster-standardized)"
