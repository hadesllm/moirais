# moirais.fn — function file (hadesllm/moirais)
"""Gradient boosting residual-fitting step (regression with squared loss)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_gradient_boosting_residual"]


def geron_gradient_boosting_residual(X, y, F_prev):
    """
    Gradient boosting residual-fitting step (regression with squared loss)

    Formula: r_i^(m) = y_i - F_{m-1}(x_i); train h_m on {(x_i, r_i^(m))}

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    F_prev : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: residuals

    References
    ----------
    Géron Ch 6, Gradient Boosting section
    """
    y = np.asarray(y, dtype=float)
    n = int(y) if y.ndim == 0 else len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Gradient boosting residual-fitting step (regression with squared loss)"})


def cheatsheet():
    return "grgbm: Gradient boosting residual-fitting step (regression with squared loss)"
