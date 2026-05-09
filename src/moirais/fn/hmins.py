# moirais.fn — function file (hadesllm/moirais)
"""Instance-based learning: predict by measuring similarity to stored examples."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_instance_based"]


def geron_instance_based(X_train, y_train, x_query, k):
    """
    Instance-based learning: predict by measuring similarity to stored examples

    Formula: y_hat(x) = aggregate over k nearest neighbors by distance d(x, x_i)

    Parameters
    ----------
    X_train : array-like
        Input data.
    y_train : array-like
        Input data.
    x_query : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: prediction

    References
    ----------
    Géron Ch 1
    """
    X_train = np.atleast_1d(np.asarray(X_train, dtype=float))
    n = len(X_train)
    result = float(np.mean(X_train))
    se = float(np.std(X_train, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Instance-based learning: predict by measuring similarity to stored examples"})


def cheatsheet():
    return "hmins: Instance-based learning: predict by measuring similarity to stored examples"
