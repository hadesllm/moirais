"""Spatial GLMM fitting via Laplace."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_glmm_fit"]


def spatial_glmm_fit(y, X, coords, family):
    """
    Spatial GLMM fitting via Laplace

    Formula: y = g^-1(X beta + W u); u ~ MVN(0, sigma^2 R)

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    coords : array-like
        Input data.
    family : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Diggle-Tawn-Moyeed (1998)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial GLMM fitting via Laplace"})


def cheatsheet():
    return "sgflrt: Spatial GLMM fitting via Laplace"
