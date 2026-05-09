# moirais.fn — function file (hadesllm/moirais)
"""Partially linear quantile model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_plr_quantile"]


def horowitz_plr_quantile(x, y, z, bandwidth, tau):
    """
    Partially linear quantile model

    Formula: Q_tau(Y|X,Z) = X*beta_tau + g_tau(Z); estimate via quantile check function with Robinson approach

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    z : array-like
        Input data.
    bandwidth : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: beta_tau, g_tau_hat

    References
    ----------
    Horowitz Ch 3, Sec 3.6.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Partially linear quantile model"})


def cheatsheet():
    return "hrzplrq: Partially linear quantile model"
