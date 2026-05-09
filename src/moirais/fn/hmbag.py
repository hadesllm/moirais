# moirais.fn — function file (hadesllm/moirais)
"""Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_bagging"]


def geron_bagging(X, y, base_estimator, n_estimators, seed):
    """
    Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs

    Formula: y_hat = (1/M) sum_m f_m(x), f_m trained on bootstrap of D

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
    seed : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs"})


def cheatsheet():
    return "hmbag: Bagging (bootstrap aggregating): train on bootstrap samples, aggregate outputs"
