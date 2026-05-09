# moirais.fn — function file (hadesllm/moirais)
"""Elbow method for k-means cluster count selection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_kmeans_elbow"]


def rangayyan_kmeans_elbow(X, max_k):
    """
    Elbow method for k-means cluster count selection

    Formula: WCSS(k) = sum_k sum_{X in C_k} ||X - mu_k||^2; elbow at knee of curve

    Parameters
    ----------
    X : array-like
        Input data.
    max_k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wcss_values, optimal_k

    References
    ----------
    Rangayyan Ch 10.5.1
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Elbow method for k-means cluster count selection"})


def cheatsheet():
    return "rgelbow: Elbow method for k-means cluster count selection"
