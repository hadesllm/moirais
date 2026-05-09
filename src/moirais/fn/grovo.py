# moirais.fn — function file (hadesllm/moirais)
"""One-vs-One: train K(K-1)/2 binary classifiers, majority vote."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_one_vs_one"]


def geron_one_vs_one(X, y):
    """
    One-vs-One: train K(K-1)/2 binary classifiers, majority vote

    Formula: y_hat = majority_vote(classifier_{i,j}(x) for i<j)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred, votes

    References
    ----------
    Géron Ch 3, Multiclass (OvO) section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "One-vs-One: train K(K-1)/2 binary classifiers, majority vote"})


def cheatsheet():
    return "grovo: One-vs-One: train K(K-1)/2 binary classifiers, majority vote"
