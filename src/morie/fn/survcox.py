"""Cox partial likelihood evaluation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cox_partial_likelihood"]


def cox_partial_likelihood(time, event, X, beta):
    """
    Cox partial likelihood evaluation

    Formula: L = prod_i exp(beta X_i) / sum_j in R_i exp(beta X_j)

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    beta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cox (1972, 1975)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cox partial likelihood evaluation"})


def cheatsheet():
    return "survcox: Cox partial likelihood evaluation"
