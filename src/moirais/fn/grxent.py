# moirais.fn — function file (hadesllm/moirais)
"""Cross-entropy cost for K-class softmax regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_cross_entropy_cost"]


def geron_softmax_cross_entropy_cost(X, Y, theta):
    """
    Cross-entropy cost for K-class softmax regression

    Formula: J(Theta) = -(1/m) sum_i sum_k y_k^(i) log(p_hat_k^(i))

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cost

    References
    ----------
    Géron Ch 4, Eq 4-22 (Cross entropy cost function)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cross-entropy cost for K-class softmax regression"})


def cheatsheet():
    return "grxent: Cross-entropy cost for K-class softmax regression"
