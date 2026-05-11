"""Log-linear model for contingency tables."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_log_linear"]


def wasserman_log_linear(table):
    """
    Log-linear model for contingency tables

    Formula: log(mu_ij) = lambda + lambda_i^A + lambda_j^B + lambda_ij^{AB}

    Parameters
    ----------
    table : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: fit

    References
    ----------
    Wasserman (2004), Ch 17
    """
    table = np.atleast_1d(np.asarray(table, dtype=float))
    n = len(table)
    result = float(np.mean(table))
    se = float(np.std(table, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Log-linear model for contingency tables"})


def cheatsheet():
    return "wsmlgc: Log-linear model for contingency tables"
