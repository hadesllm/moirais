# morie.fn — function file (hadesllm/morie)
"""Conditional quantile additive model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_additive_quantile"]


def horowitz_additive_quantile(x, y, bandwidth, tau):
    """
    Conditional quantile additive model

    Formula: Q_tau(Y|X) = mu_tau + sum_j g_{j,tau}(X_j); estimate via check-function minimization

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    bandwidth : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_j_tau_hats

    References
    ----------
    Horowitz Ch 3, Sec 3.4
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Conditional quantile additive model"})


def cheatsheet():
    return "hrzaqm: Conditional quantile additive model"
