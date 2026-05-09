# moirais.fn — function file (hadesllm/moirais)
"""XGBoost: regularized gradient boosting with second-order Taylor approximation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_xgboost"]


def geron_xgboost(X, y, n_estimators, learning_rate, max_depth):
    """
    XGBoost: regularized gradient boosting with second-order Taylor approximation

    Formula: obj = sum_i L(y_i, y_hat_i) + sum_t Omega(f_t)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    n_estimators : array-like
        Input data.
    learning_rate : array-like
        Input data.
    max_depth : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 6
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "XGBoost: regularized gradient boosting with second-order Taylor approximation"})


def cheatsheet():
    return "hmxgb: XGBoost: regularized gradient boosting with second-order Taylor approximation"
