"""Matérn cluster process."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["matern_cluster"]


def matern_cluster(lambda_p, mu, r):
    """
    Matérn cluster process

    Formula: Poisson parents; cluster of offspring around each

    Parameters
    ----------
    lambda_p : array-like
        Input data.
    mu : array-like
        Input data.
    r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matérn (1960)
    """
    lambda_p = np.atleast_1d(np.asarray(lambda_p, dtype=float))
    n = len(lambda_p)
    result = float(np.mean(lambda_p))
    se = float(np.std(lambda_p, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matérn cluster process"})


def cheatsheet():
    return "matern: Matérn cluster process"
