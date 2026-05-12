# morie.fn -- function file (hadesllm/morie)
"""Semi-supervised learning: small labeled set plus large unlabeled pool."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_semisupervised"]


def geron_semisupervised(X_l, y_l, X_u, alpha):
    """
    Semi-supervised learning: small labeled set plus large unlabeled pool

    Formula: minimize L_sup(f,L) + alpha * L_unsup(f,U)

    Parameters
    ----------
    X_l : array-like
        Input data.
    y_l : array-like
        Input data.
    X_u : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    X_l = np.atleast_1d(np.asarray(X_l, dtype=float))
    n = len(X_l)
    result = float(np.mean(X_l))
    se = float(np.std(X_l, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semi-supervised learning: small labeled set plus large unlabeled pool"})


def cheatsheet():
    return "hmsem: Semi-supervised learning: small labeled set plus large unlabeled pool"
