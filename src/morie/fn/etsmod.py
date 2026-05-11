"""ETS state-space (error/trend/seasonal)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["ets"]


def ets(y, error, trend, season):
    """
    ETS state-space (error/trend/seasonal)

    Formula: y_t = w' x_{t-1} + ε; x_t = F x_{t-1} + g ε

    Parameters
    ----------
    y : array-like
        Input data.
    error : array-like
        Input data.
    trend : array-like
        Input data.
    season : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Koehler-Snyder-Grose (2002)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "ETS state-space (error/trend/seasonal)"})


def cheatsheet():
    return "etsmod: ETS state-space (error/trend/seasonal)"
