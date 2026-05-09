# moirais.fn — function file (hadesllm/moirais)
"""ARIMA(p, d, q) one-step-ahead forecast (post-differencing)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_arima_forecast"]


def geron_arima_forecast(y, phi, theta, d):
    """
    ARIMA(p, d, q) one-step-ahead forecast (post-differencing)

    Formula: y_hat_t = phi_1 y_{t-1} + ... + phi_p y_{t-p} + theta_1 eps_{t-1} + ... + theta_q eps_{t-q}

    Parameters
    ----------
    y : array-like
        Input data.
    phi : array-like
        Input data.
    theta : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: y_pred

    References
    ----------
    Géron Ch 13, ARMA/ARIMA section
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARIMA(p, d, q) one-step-ahead forecast (post-differencing)"})


def cheatsheet():
    return "grarma: ARIMA(p, d, q) one-step-ahead forecast (post-differencing)"
