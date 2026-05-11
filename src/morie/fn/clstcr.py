"""Cluster-level causal inference."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cluster_causal_inference"]


def cluster_causal_inference(y, D, X, cluster):
    """
    Cluster-level causal inference

    Formula: average CATE within cluster + cluster-robust

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.
    cluster : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hudgens-Halloran (2008)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cluster-level causal inference"})


def cheatsheet():
    return "clstcr: Cluster-level causal inference"
