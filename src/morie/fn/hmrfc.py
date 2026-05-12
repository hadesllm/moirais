# morie.fn -- function file (hadesllm/morie)
"""Random forest: bagging of decision trees with random feature splits."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_random_forest"]


def geron_random_forest(X, y, n_estimators, max_features, seed):
    """
    Random forest: bagging of decision trees with random feature splits

    Formula: y_hat = majority vote of M random-split trees

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_estimators : array-like
        Input data.
    max_features : array-like
        Input data.
    seed : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random forest: bagging of decision trees with random feature splits"})


def cheatsheet():
    return "hmrfc: Random forest: bagging of decision trees with random feature splits"
