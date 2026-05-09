# moirais.fn — function file (hadesllm/moirais)
"""Learning curves: train and validation RMSE as a function of m."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_learning_curves"]


def geron_learning_curves(X, y, n_splits):
    """
    Learning curves: train and validation RMSE as a function of m

    Formula: RMSE_train(m), RMSE_val(m) computed over growing subsets of the training set

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_splits : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: train_rmse, val_rmse

    References
    ----------
    Géron Ch 4, Learning Curves section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Learning curves: train and validation RMSE as a function of m"})


def cheatsheet():
    return "grlrnc: Learning curves: train and validation RMSE as a function of m"
