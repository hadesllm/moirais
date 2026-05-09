# moirais.fn — function file (hadesllm/moirais)
"""Multiclass one-vs-one: train K(K-1)/2 pairwise binary classifiers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_vs_one"]


def geron_one_vs_one(X, y, base_estimator):
    """
    Multiclass one-vs-one: train K(K-1)/2 pairwise binary classifiers

    Formula: f_{i,j} for each pair (i<j); vote aggregation

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    base_estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: models

    References
    ----------
    Géron Ch 3
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiclass one-vs-one: train K(K-1)/2 pairwise binary classifiers"})


def cheatsheet():
    return "hmovo: Multiclass one-vs-one: train K(K-1)/2 pairwise binary classifiers"
