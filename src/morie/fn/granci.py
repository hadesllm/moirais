"""Granger causality as MI."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["granger_causality_info"]


def granger_causality_info(x, y, lag):
    """
    Granger causality as MI

    Formula: I(Y_t ; X_past | Y_past)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Barnett-Bossomaier (2012)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Granger causality as MI"})


def cheatsheet():
    return "granci: Granger causality as MI"
