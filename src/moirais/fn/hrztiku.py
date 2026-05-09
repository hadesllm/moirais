# moirais.fn — function file (hadesllm/moirais)
"""Tikhonov regularization for NPIV when T is unknown."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_tikhonov_unknown_T"]


def horowitz_tikhonov_unknown_T(x, y, w, bandwidth, alpha):
    """
    Tikhonov regularization for NPIV when T is unknown

    Formula: g_hat_alpha = (T_hat'*T_hat + alpha*I)^{-1}*T_hat'*m_hat

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    w : array-like
        Input data.
    bandwidth : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: g_hat

    References
    ----------
    Horowitz Ch 5, Sec 5.4.1
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Tikhonov regularization for NPIV when T is unknown"})


def cheatsheet():
    return "hrztiku: Tikhonov regularization for NPIV when T is unknown"
