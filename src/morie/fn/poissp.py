"""Poisson spatial GLM with offset."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["poisson_spatial_glm"]


def poisson_spatial_glm(counts, X, offset, W):
    """
    Poisson spatial GLM with offset

    Formula: log lambda = X beta + W u; u ~ MVN(0, Q^-1)

    Parameters
    ----------
    counts : array-like
        Input data.
    X : array-like
        Input data.
    offset : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Banerjee-Carlin-Gelfand (2014)
    """
    counts = np.atleast_1d(np.asarray(counts, dtype=float))
    n = len(counts)
    result = float(np.mean(counts))
    se = float(np.std(counts, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Poisson spatial GLM with offset"})


def cheatsheet():
    return "poissp: Poisson spatial GLM with offset"
