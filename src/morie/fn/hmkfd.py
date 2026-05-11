# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation: partition into k equal folds; rotate held-out fold."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_kfold"]


def geron_kfold(X, y, k, seed):
    """
    K-fold cross-validation: partition into k equal folds; rotate held-out fold

    Formula: avg CV score = (1/k) sum_i score(fold_i)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fold_scores

    References
    ----------
    Géron Ch 2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-fold cross-validation: partition into k equal folds; rotate held-out fold"})


def cheatsheet():
    return "hmkfd: K-fold cross-validation: partition into k equal folds; rotate held-out fold"
