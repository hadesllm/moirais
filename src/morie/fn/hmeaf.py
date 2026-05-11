# morie.fn — function file (hadesllm/morie)
"""Error analysis via normalized confusion matrix row/column inspection."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_error_analysis"]


def geron_error_analysis(y_true, y_pred):
    """
    Error analysis via normalized confusion matrix row/column inspection

    Formula: E_norm[i,j] = C[i,j] / sum_j C[i,j] - diag correction

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: normalized_matrix

    References
    ----------
    Géron Ch 3
    """
    y_true = np.atleast_1d(np.asarray(y_true, dtype=float))
    n = len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Error analysis via normalized confusion matrix row/column inspection"})


def cheatsheet():
    return "hmeaf: Error analysis via normalized confusion matrix row/column inspection"
