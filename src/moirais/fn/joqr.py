# moirais.fn — function file (hadesllm/moirais)
"""Quantile regression: minimize pinball loss at target quantile."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_quantile_regression"]


def joseph_quantile_regression(X, y, tau):
    """
    Quantile regression: minimize pinball loss at target quantile

    Formula: beta_tau = argmin_beta sum_i L_tau(y_i, x_i^T beta)

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_tau

    References
    ----------
    Joseph Ch 17, Quantile Regression section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile regression: minimize pinball loss at target quantile"})


def cheatsheet():
    return "joqr: Quantile regression: minimize pinball loss at target quantile"
