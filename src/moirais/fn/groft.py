# moirais.fn — function file (hadesllm/moirais)
"""Overfitting gap: training accuracy minus validation accuracy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_overfitting_gap"]


def geron_overfitting_gap(train_scores, val_scores):
    """
    Overfitting gap: training accuracy minus validation accuracy

    Formula: gap = acc_train - acc_val (positive gap = overfitting)

    Parameters
    ----------
    train_scores : array-like
        Input data.
    val_scores : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: gap

    References
    ----------
    Géron Ch 1, Overfitting section
    """
    train_scores = np.asarray(train_scores, dtype=float)
    n = int(train_scores) if train_scores.ndim == 0 else len(train_scores)
    result = float(np.mean(train_scores))
    se = float(np.std(train_scores, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Overfitting gap: training accuracy minus validation accuracy"})


def cheatsheet():
    return "groft: Overfitting gap: training accuracy minus validation accuracy"
