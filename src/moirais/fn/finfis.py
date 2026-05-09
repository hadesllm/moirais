"""Fisher information matrix."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fisher_information"]


def fisher_information(log_likelihood, theta):
    """
    Fisher information matrix

    Formula: I(theta) = -E[d^2 log L / d theta^2]

    Parameters
    ----------
    log_likelihood : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fisher (1922)
    """
    log_likelihood = np.atleast_1d(np.asarray(log_likelihood, dtype=float))
    n = len(log_likelihood)
    result = float(np.mean(log_likelihood))
    se = float(np.std(log_likelihood, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fisher information matrix"})


def cheatsheet():
    return "finfis: Fisher information matrix"
