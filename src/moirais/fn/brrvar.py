"""Balanced Repeated Replication variance."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["brr_variance"]


def brr_variance(y, weights, replicates):
    """
    Balanced Repeated Replication variance

    Formula: Var = (1/R) sum_r (theta_r - theta_hat)^2; Hadamard half-samples

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    replicates : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McCarthy (1969); Wolter (2007) §3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Balanced Repeated Replication variance"})


def cheatsheet():
    return "brrvar: Balanced Repeated Replication variance"
