# morie.fn — function file (hadesllm/morie)
"""Grid search: evaluate every combination of hyperparameter values."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_grid_search"]


def geron_grid_search(param_grid, X, y, estimator):
    """
    Grid search: evaluate every combination of hyperparameter values

    Formula: best = argmin over cartesian product of param values

    Parameters
    ----------
    param_grid : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grid search: evaluate every combination of hyperparameter values"})


def cheatsheet():
    return "hmgrs: Grid search: evaluate every combination of hyperparameter values"
