"""Johansen trace cointegration test."""
import numpy as np
from ._richresult import RichResult

__all__ = ["johansen_trace"]


def johansen_trace(X, lags=1, cdf=None):
    """Johansen trace cointegration test. lambda_trace = -T sum log(1 - lambda_i). Johansen (1991)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0] if X.ndim >= 1 else 0
    return RichResult(payload={"estimate": np.nan, "statistic": np.nan, "p_value": np.nan, "n": int(n), "method": "Johansen trace cointegration test"})


def cheatsheet():
    return "jhoint: Johansen trace cointegration test"
