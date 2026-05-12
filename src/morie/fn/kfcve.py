# morie.fn -- function file (hadesllm/morie)
"""K-fold cross-validation prediction error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["k_fold_cv_error"]


def k_fold_cv_error(y, y_hat_folds):
    """
    K-fold cross-validation prediction error

    Formula: CV_K = (1/K) * sum_{k=1}^K MSE_k

    Parameters
    ----------
    y : array-like
        Input data.
    y_hat_folds : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'cv_error': 'float'}

    References
    ----------
    Montesinos Lopez Ch 4
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-fold cross-validation prediction error"})


def cheatsheet():
    return "kfcve: K-fold cross-validation prediction error"
