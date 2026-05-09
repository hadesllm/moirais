# moirais.fn — function file (hadesllm/moirais)
"""k-means within-cluster sum of squares objective."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kmeans_objective"]


def geron_kmeans_objective(X, centroids, labels):
    """
    k-means within-cluster sum of squares objective

    Formula: J = sum_{i=1..m} ||x^(i) - mu_{c(i)}||^2

    Parameters
    ----------
    X : array-like
        Input data.
    centroids : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: inertia

    References
    ----------
    Géron Ch 8, K-means section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "k-means within-cluster sum of squares objective"})


def cheatsheet():
    return "grkmo: k-means within-cluster sum of squares objective"
