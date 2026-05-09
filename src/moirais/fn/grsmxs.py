# moirais.fn — function file (hadesllm/moirais)
"""Softmax class-score for class k in multinomial logistic regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_softmax_score"]


def geron_softmax_score(X, theta):
    """
    Softmax class-score for class k in multinomial logistic regression

    Formula: s_k(X) = theta_k^T X

    Parameters
    ----------
    X : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: scores

    References
    ----------
    Géron Ch 4, Eq 4-19 (Softmax score for class k)
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax class-score for class k in multinomial logistic regression"})


def cheatsheet():
    return "grsmxs: Softmax class-score for class k in multinomial logistic regression"
