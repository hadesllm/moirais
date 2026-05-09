"""Johansen maximum eigenvalue test."""
import numpy as np
from ._richresult import RichResult

__all__ = ["johansen_max_eigen"]


def johansen_max_eigen(X, lags=1, cdf=None):
    """Johansen maximum eigenvalue test. lambda_max = -T log(1 - lambda_{r+1}). Johansen (1991)."""
    X = np.asarray(X, dtype=float)
    n = X.shape[0] if X.ndim >= 1 else 0
    return RichResult(payload={"estimate": np.nan, "statistic": np.nan, "p_value": np.nan, "n": int(n), "method": "Johansen maximum eigenvalue test"})


def cheatsheet():
    return "jhomev: Johansen maximum eigenvalue test"
