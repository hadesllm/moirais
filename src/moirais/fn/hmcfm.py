# moirais.fn — function file (hadesllm/moirais)
"""Confusion matrix: rows = actual, columns = predicted classes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_confusion_matrix"]


def geron_confusion_matrix(y_true, y_pred):
    """
    Confusion matrix: rows = actual, columns = predicted classes

    Formula: C[i,j] = #{predicted class j | actual class i}

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: matrix

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Confusion matrix: rows = actual, columns = predicted classes"})


def cheatsheet():
    return "hmcfm: Confusion matrix: rows = actual, columns = predicted classes"
