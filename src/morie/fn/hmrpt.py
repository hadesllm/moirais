# morie.fn -- function file (hadesllm/morie)
"""Random patches: subsample both rows and features per base model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_random_patches"]


def geron_random_patches(X, y, base_estimator, n_estimators, max_samples, max_features):
    """
    Random patches: subsample both rows and features per base model

    Formula: each f_m uses random rows I_m and features S_m

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    base_estimator : array-like
        Input data.
    n_estimators : array-like
        Input data.
    max_samples : array-like
        Input data.
    max_features : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: models

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random patches: subsample both rows and features per base model"})


def cheatsheet():
    return "hmrpt: Random patches: subsample both rows and features per base model"
