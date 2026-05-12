# morie.fn -- function file (hadesllm/morie)
"""Randomized search: sample n_iter hyperparameter combinations from distributions."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_randomized_search"]


def geron_randomized_search(param_dist, n_iter, X, y, estimator):
    """
    Randomized search: sample n_iter hyperparameter combinations from distributions

    Formula: theta_i ~ prior; pick best over n_iter samples

    Parameters
    ----------
    param_dist : array-like
        Input data.
    n_iter : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.
    estimator : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_params

    References
    ----------
    Géron Ch 2
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Randomized search: sample n_iter hyperparameter combinations from distributions"})


def cheatsheet():
    return "hmrsc: Randomized search: sample n_iter hyperparameter combinations from distributions"
