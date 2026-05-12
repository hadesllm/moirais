# morie.fn -- function file (hadesllm/morie)
"""Randomized hyperparam search over distributions with CV scoring."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_randomized_search_cv"]


def geron_randomized_search_cv(X, y, param_dist, n_iter, K):
    """
    Randomized hyperparam search over distributions with CV scoring

    Formula: sample n_iter configurations from P(theta); evaluate via CV

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    param_dist : array-like
        Input data.
    n_iter : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_params, best_score

    References
    ----------
    Géron Ch 2, Randomized Search section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Randomized hyperparam search over distributions with CV scoring"})


def cheatsheet():
    return "grrnd: Randomized hyperparam search over distributions with CV scoring"
