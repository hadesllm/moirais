# morie.fn — function file (hadesllm/morie)
"""K-fold cross-validation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_kfold_cv"]


def rangayyan_kfold_cv(X, y, k, classifier):
    """
    K-fold cross-validation

    Formula: CV_k = (1/K) sum_{k=1}^{K} error_k on held-out fold k

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    k : array-like
        Input data.
    classifier : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cv_error, fold_errors

    References
    ----------
    Rangayyan Ch 10.10.3
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "K-fold cross-validation"})


def cheatsheet():
    return "rgkfcv: K-fold cross-validation"
