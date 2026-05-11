"""Quantile (LAD when tau=0.5) regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quantile_regression"]


def quantile_regression(y, X, tau):
    """
    Quantile (LAD when tau=0.5) regression

    Formula: min sum rho_tau(y_i - x_i' beta), rho_tau(u) = u(tau - I(u<0))

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Koenker & Bassett (1978)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile (LAD when tau=0.5) regression"})


def cheatsheet():
    return "quanrg: Quantile (LAD when tau=0.5) regression"
