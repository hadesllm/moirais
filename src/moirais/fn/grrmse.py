# moirais.fn — function file (hadesllm/moirais)
"""Root mean squared error — L2 norm of prediction residuals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_rmse"]


def geron_rmse(y_true, y_pred):
    """
    Root mean squared error — L2 norm of prediction residuals

    Formula: RMSE = sqrt((1/m) sum_{i=1..m} (h(x^(i)) - y^(i))^2)

    Parameters
    ----------
    y_true : array-like
        Input data.
    y_pred : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rmse

    References
    ----------
    Géron Ch 2, Eq 2-1 (Root Mean Squared Error)
    """
    y_true = np.asarray(y_true, dtype=float)
    n = int(y_true) if y_true.ndim == 0 else len(y_true)
    result = float(np.mean(y_true))
    se = float(np.std(y_true, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Root mean squared error — L2 norm of prediction residuals"})


def cheatsheet():
    return "grrmse: Root mean squared error — L2 norm of prediction residuals"
