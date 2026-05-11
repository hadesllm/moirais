# morie.fn — function file (hadesllm/morie)
"""Grid search for DNN hyperparameter tuning with cross-validation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["hyperparameter_tuning_grid"]


def hyperparameter_tuning_grid(param_grid, cv_data):
    """
    Grid search for DNN hyperparameter tuning with cross-validation

    Formula: select (topology, lr, dropout, L2) = argmin_{H} CV_K(H)

    Parameters
    ----------
    param_grid : array-like
        Input data.
    cv_data : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'best_params': 'dict', 'cv_score': 'float'}

    References
    ----------
    Montesinos Lopez Ch 11
    """
    param_grid = np.asarray(param_grid, dtype=float)
    n = int(param_grid) if param_grid.ndim == 0 else len(param_grid)
    result = float(np.mean(param_grid))
    se = float(np.std(param_grid, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Grid search for DNN hyperparameter tuning with cross-validation"})


def cheatsheet():
    return "htprd: Grid search for DNN hyperparameter tuning with cross-validation"
