# moirais.fn — function file (hadesllm/moirais)
"""Unsupervised learning: discover structure from unlabeled data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_unsupervised_learning"]


def geron_unsupervised_learning(X):
    """
    Unsupervised learning: discover structure from unlabeled data

    Formula: maximize p_theta(X) or cluster assignments

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unsupervised learning: discover structure from unlabeled data"})


def cheatsheet():
    return "hmuns: Unsupervised learning: discover structure from unlabeled data"
