# morie.fn — function file (hadesllm/morie)
"""Mean absolute error — L1 norm of prediction residuals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_mae"]


def geron_mae(y_true, y_pred):
    """
    Mean absolute error — L1 norm of prediction residuals

    Formula: MAE = (1/m) sum_{i=1..m} |h(x^(i)) - y^(i)|

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mae

    References
    ----------
    Géron Ch 2, Eq 2-2 (Mean Absolute Error)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mean absolute error — L1 norm of prediction residuals"})


def cheatsheet():
    return "grmae: Mean absolute error — L1 norm of prediction residuals"
