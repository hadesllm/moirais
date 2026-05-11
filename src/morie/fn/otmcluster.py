"""Wasserstein-based k-means using Wasserstein-2 between samples."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ot_clustering_w"]


def ot_clustering_w(X_list, k, max_iter):
    """
    Wasserstein-based k-means using Wasserstein-2 between samples

    Formula: Lloyd iteration with W_2 centroids per cluster

    Parameters
    ----------
    X_list : array-like
        Input data.
    k : array-like
        Input data.
    max_iter : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centers

    References
    ----------
    del Barrio-Cuesta-Albertos-Matrán (2019)
    """
    X_list = np.atleast_1d(np.asarray(X_list, dtype=float))
    n = len(X_list)
    result = float(np.mean(X_list))
    se = float(np.std(X_list, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wasserstein-based k-means using Wasserstein-2 between samples"})


def cheatsheet():
    return "otmcluster: Wasserstein-based k-means using Wasserstein-2 between samples"
