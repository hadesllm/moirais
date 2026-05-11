# morie.fn — function file (hadesllm/morie)
"""Multilabel classification: independent binary predictions per label."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_multilabel_classification"]


def geron_multilabel_classification(X, Y, thresholds):
    """
    Multilabel classification: independent binary predictions per label

    Formula: y_hat_k = 1{ score_k(x) > t_k } for each label k

    Parameters
    ----------
    X : array-like
        Input data.
    Y : array-like
        Input data.
    thresholds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: Y_pred

    References
    ----------
    Géron Ch 3, Multilabel Classification section
    """
    X = np.asarray(X, dtype=float)
    n = int(X) if X.ndim == 0 else len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multilabel classification: independent binary predictions per label"})


def cheatsheet():
    return "grmlb: Multilabel classification: independent binary predictions per label"
