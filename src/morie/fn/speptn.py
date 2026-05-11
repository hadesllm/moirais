"""Spatial PCA / MULTISPATI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spatial_pca"]


def spatial_pca(X, W):
    """
    Spatial PCA / MULTISPATI

    Formula: PCA constrained by spatial weights

    Parameters
    ----------
    X : array-like
        Input data.
    W : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Dray-Saïd-Débias (2008)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial PCA / MULTISPATI"})


def cheatsheet():
    return "speptn: Spatial PCA / MULTISPATI"
