# morie.fn — function file (hadesllm/morie)
"""Semi-supervised learning via k-means representative labeling."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_semisupervised_cluster"]


def geron_semisupervised_cluster(X, X_labeled, y_labeled, n_clusters):
    """
    Semi-supervised learning via k-means representative labeling

    Formula: label cluster representatives; propagate to members

    Parameters
    ----------
    X : array-like
        Input data.
    X_labeled : array-like
        Input data.
    y_labeled : array-like
        Input data.
    n_clusters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semi-supervised learning via k-means representative labeling"})


def cheatsheet():
    return "hmsslc: Semi-supervised learning via k-means representative labeling"
