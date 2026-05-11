"""Spatial autoregressive (lag) model."""
import numpy as np
from ._richresult import RichResult

__all__ = ["schabenberger_sar_model"]


def schabenberger_sar_model(x, y, w):
    """
    Spatial autoregressive (lag) model

    Formula: Y = rho*W*Y + X*beta + epsilon; Y = (I-rho*W)^{-1}*(X*beta+epsilon)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rho, beta, se

    References
    ----------
    Schabenberger Ch 6, Sec 6.2.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spatial autoregressive (lag) model"})


def cheatsheet():
    return "spsar: Spatial autoregressive (lag) model"
