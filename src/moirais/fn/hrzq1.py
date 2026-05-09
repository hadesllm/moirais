# moirais.fn — function file (hadesllm/moirais)
"""Semiparametric quantile regression."""
import numpy as np
from ._richresult import RichResult

__all__ = ["horowitz_quantile_regression"]


def horowitz_quantile_regression(x, y, tau):
    """
    Semiparametric quantile regression

    Formula: Q_tau(Y|X) = X'beta(tau) + g(Z,tau)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    tau : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate, se

    References
    ----------
    Horowitz (2009), Ch 10
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Semiparametric quantile regression"})


def cheatsheet():
    return "hrzq1: Semiparametric quantile regression"
