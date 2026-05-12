# morie.fn -- function file (hadesllm/morie)
"""Batch (offline) learning: train once on full dataset, then deploy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_batch_learning"]


def geron_batch_learning(X, y):
    """
    Batch (offline) learning: train once on full dataset, then deploy

    Formula: theta = argmin_theta L(theta; D_train)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 1
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Batch (offline) learning: train once on full dataset, then deploy"})


def cheatsheet():
    return "hmbat: Batch (offline) learning: train once on full dataset, then deploy"
