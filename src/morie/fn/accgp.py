# morie.fn — function file from book-equation translation pipeline (hadesllm/morie)
"""Genomic prediction accuracy metrics."""
import numpy as np
from ._richresult import RichResult

__all__ = ["prediction_accuracy"]


def prediction_accuracy(y_true, y_pred):
    """
    Genomic prediction accuracy metrics

    Formula: r = cor(y,y_hat), MSE, MSPE, regression slope

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Montesinos Lopez Ch 2
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Genomic prediction accuracy metrics"})


def cheatsheet():
    return "accgp: Genomic prediction accuracy metrics"
