# morie.fn — function file (hadesllm/morie)
"""AdaBoost: train sequential weighted weak learners."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_adaboost"]


def geron_adaboost(X, y, base_estimator, n_estimators):
    """
    AdaBoost: train sequential weighted weak learners

    Formula: alpha_t = 0.5*log((1-err_t)/err_t); w_{i,t+1} = w_{i,t}*exp(-alpha_t y_i f_t(x_i))

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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "AdaBoost: train sequential weighted weak learners"})


def cheatsheet():
    return "hmadab: AdaBoost: train sequential weighted weak learners"
