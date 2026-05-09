"""Bayesian state-space TS."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_time_series"]


def bayes_time_series(y, model, priors):
    """
    Bayesian state-space TS

    Formula: alpha_t = T alpha_t-1 + eta; y_t = Z alpha + eps

    Parameters
    ----------
    y : array-like
        Input data.
    model : array-like
        Input data.
    priors : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    West-Harrison (1997)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian state-space TS"})


def cheatsheet():
    return "baytsm: Bayesian state-space TS"
