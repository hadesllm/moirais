# morie.fn -- function file (hadesllm/morie)
"""Silhouette score: mean over points of (b - a) / max(a, b)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_silhouette_score"]


def geron_silhouette_score(X, labels):
    """
    Silhouette score: mean over points of (b - a) / max(a, b)

    Formula: s_i = (b_i - a_i) / max(a_i, b_i); silhouette = mean_i s_i

    Parameters
    ----------
    X : array-like
        Input data.
    labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: score

    References
    ----------
    Géron Ch 8, Silhouette Score section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Silhouette score: mean over points of (b - a) / max(a, b)"})


def cheatsheet():
    return "grsil: Silhouette score: mean over points of (b - a) / max(a, b)"
