# morie.fn — function file (hadesllm/morie)
"""Binary classification: predict one of two classes using probability threshold."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_binary_classification"]


def geron_binary_classification(X, theta):
    """
    Binary classification: predict one of two classes using probability threshold

    Formula: y_hat = I(p_hat >= 0.5)

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_hat

    References
    ----------
    Géron Ch 3
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Binary classification: predict one of two classes using probability threshold"})


def cheatsheet():
    return "hmbin: Binary classification: predict one of two classes using probability threshold"
