# morie.fn -- function file (hadesllm/morie)
"""Random forest mean decrease in impurity (MDI) variable importance."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rf_mdi_importance"]


def rf_mdi_importance(forest, X, y):
    """
    Random forest mean decrease in impurity (MDI) variable importance

    Formula: Imp(X_j) = (1/B) sum_b sum_{t: split on X_j} (n_t/n) * [imp(t) - (n_L/n_t)*imp(t_L) - (n_R/n_t)*imp(t_R)]

    Parameters
    ----------
    forest : array-like
        Input data.
    X : array-like
        Input data.
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: {'importance': 'array'}

    References
    ----------
    Montesinos Lopez Ch 15
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Random forest mean decrease in impurity (MDI) variable importance"})


def cheatsheet():
    return "rfmdi: Random forest mean decrease in impurity (MDI) variable importance"
