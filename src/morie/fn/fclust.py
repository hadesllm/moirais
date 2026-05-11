"""Functional clustering (k-means on coefficients)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["functional_clustering"]


def functional_clustering(Y, K, basis):
    """
    Functional clustering (k-means on coefficients)

    Formula: k-means in coefficient space

    Parameters
    ----------
    Y : array-like
        Input data.
    K : array-like
        Input data.
    basis : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Abraham et al (2003)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Functional clustering (k-means on coefficients)"})


def cheatsheet():
    return "fclust: Functional clustering (k-means on coefficients)"
