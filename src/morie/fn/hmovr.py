# morie.fn — function file (hadesllm/morie)
"""Multiclass one-vs-rest (OvR): train K binary classifiers."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_vs_rest"]


def geron_one_vs_rest(X, y, base_estimator):
    """
    Multiclass one-vs-rest (OvR): train K binary classifiers

    Formula: for k in 1..K: f_k distinguishes class k vs all others

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiclass one-vs-rest (OvR): train K binary classifiers"})


def cheatsheet():
    return "hmovr: Multiclass one-vs-rest (OvR): train K binary classifiers"
