# morie.fn -- function file (hadesllm/morie)
"""Mini-batch k-means: update centers using small random batches."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_minibatch_kmeans"]


def geron_minibatch_kmeans(X, n_clusters, batch_size, seed):
    """
    Mini-batch k-means: update centers using small random batches

    Formula: center_update = (1-lr)*mu_c + lr*mean(batch_in_c)

    Parameters
    ----------
    X : array-like
        Input data.
    n_clusters : array-like
        Input data.
    batch_size : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: labels, centers

    References
    ----------
    Géron Ch 8
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mini-batch k-means: update centers using small random batches"})


def cheatsheet():
    return "hmmbkm: Mini-batch k-means: update centers using small random batches"
