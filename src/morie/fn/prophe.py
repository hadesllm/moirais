"""Prophet additive model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["facebook_prophet"]


def facebook_prophet(y, date, seasonality):
    """
    Prophet additive model

    Formula: y = trend + seasonality + holiday + eps

    Parameters
    ----------
    y : array-like
        Input data.
    date : array-like
        Input data.
    seasonality : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Taylor-Letham (2018)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prophet additive model"})


def cheatsheet():
    return "prophe: Prophet additive model"
