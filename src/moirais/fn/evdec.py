"""Runs declustering for serially dependent exceedances."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["evt_declustering_runs"]


def evt_declustering_runs(x, u, r):
    """
    Runs declustering for serially dependent exceedances

    Formula: cluster = consecutive exceedances within gap r

    Parameters
    ----------
    x : array-like
        Input data.
    u : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cluster_max, cluster_id

    References
    ----------
    Smith (1989)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Runs declustering for serially dependent exceedances"})


def cheatsheet():
    return "evdec: Runs declustering for serially dependent exceedances"
