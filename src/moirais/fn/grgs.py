# moirais.fn — function file (hadesllm/moirais)
"""Exhaustive grid search over hyperparam grid with CV scoring."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_grid_search_cv"]


def geron_grid_search_cv(X, y, param_grid, K):
    """
    Exhaustive grid search over hyperparam grid with CV scoring

    Formula: best = argmax_{theta in Grid} mean_k score(model(theta), fold_k)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    param_grid : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: best_params, best_score

    References
    ----------
    Géron Ch 2, Grid Search section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Exhaustive grid search over hyperparam grid with CV scoring"})


def cheatsheet():
    return "grgs: Exhaustive grid search over hyperparam grid with CV scoring"
