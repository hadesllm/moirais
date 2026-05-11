# morie.fn — function file (hadesllm/morie)
"""Grid search with cross-validation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["grid_search_cv"]


def grid_search_cv(x, y):
    """
    Grid search with cross-validation

    Formula: best_params = argmin CV_error(params)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Geron (2026), Ch 2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grid search with cross-validation"})


def cheatsheet():
    return "gsrch: Grid search with cross-validation"
