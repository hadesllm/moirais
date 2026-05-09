# moirais.fn — function file (hadesllm/moirais)
"""Overfitting: training error much lower than validation error."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_overfitting"]


def geron_overfitting(train_err, val_err):
    """
    Overfitting: training error much lower than validation error

    Formula: E_train << E_val; gap = E_val - E_train

    Parameters
    ----------
    train_err : array-like
        Input data.
    val_err : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: overfit_gap

    References
    ----------
    Géron Ch 1
    """
    train_err = np.atleast_1d(np.asarray(train_err, dtype=float))
    n = len(train_err)
    result = float(np.mean(train_err))
    se = float(np.std(train_err, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Overfitting: training error much lower than validation error"})


def cheatsheet():
    return "hmovf: Overfitting: training error much lower than validation error"
