# morie.fn -- function file (hadesllm/morie)
"""Underfitting: high training error because the model is too simple."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_underfitting"]


def geron_underfitting(train_err, threshold):
    """
    Underfitting: high training error because the model is too simple

    Formula: E_train high; model bias > variance

    Parameters
    ----------
    train_err : array-like
        Input data.
    threshold : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_underfit

    References
    ----------
    Géron Ch 1
    """
    train_err = np.atleast_1d(np.asarray(train_err, dtype=float))
    n = len(train_err)
    result = float(np.mean(train_err))
    se = float(np.std(train_err, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Underfitting: high training error because the model is too simple"})


def cheatsheet():
    return "hmuf: Underfitting: high training error because the model is too simple"
