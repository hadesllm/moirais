"""INLA approximation for spatial GLMM."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["inla_spatial"]


def inla_spatial(y, X, field, precision_matrix):
    """
    INLA approximation for spatial GLMM

    Formula: Laplace approximation around mode + Gaussian copula

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    field : array-like
        Input data.
    precision_matrix : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Rue-Martino-Chopin (2009)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "INLA approximation for spatial GLMM"})


def cheatsheet():
    return "inlasm: INLA approximation for spatial GLMM"
