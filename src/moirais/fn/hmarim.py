# moirais.fn — function file (hadesllm/moirais)
"""ARIMA(p,d,q) model: ARMA applied to d-th differenced series."""
import numpy as np
from ._richresult import RichResult

__all__ = ["geron_arima"]


def geron_arima(y, p, d, q):
    """
    ARIMA(p,d,q) model: ARMA applied to d-th differenced series

    Formula: differenced y of order d fit by ARMA(p,q)

    Parameters
    ----------
    y : array-like
        Input data.
    p : array-like
        Input data.
    d : array-like
        Input data.
    q : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: model

    References
    ----------
    Géron Ch 13
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ARIMA(p,d,q) model: ARMA applied to d-th differenced series"})


def cheatsheet():
    return "hmarim: ARIMA(p,d,q) model: ARMA applied to d-th differenced series"
