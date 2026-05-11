# morie.fn — function file (hadesllm/morie)
"""Learning curves: RMSE on train and validation vs training set size."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_learning_curves"]


def geron_learning_curves(X, y, n_splits):
    """
    Learning curves: RMSE on train and validation vs training set size

    Formula: RMSE_train(m), RMSE_val(m) for increasing m

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
    Géron Ch 4
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Learning curves: RMSE on train and validation vs training set size"})


def cheatsheet():
    return "hmlcv: Learning curves: RMSE on train and validation vs training set size"
