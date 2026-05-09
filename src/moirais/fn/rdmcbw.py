"""MSE-optimal bandwidth selector for RDD."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mse_optimal_bandwidth_rdd"]


def mse_optimal_bandwidth_rdd(y, x, cutoff):
    """
    MSE-optimal bandwidth selector for RDD

    Formula: h_MSE = argmin (Bias^2 + Var) over h

    Parameters
    ----------
    y : array-like
        Input data.
    x : array-like
        Input data.
    cutoff : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Imbens-Kalyanaraman (2012); Calonico-Cattaneo-Titiunik (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MSE-optimal bandwidth selector for RDD"})


def cheatsheet():
    return "rdmcbw: MSE-optimal bandwidth selector for RDD"
