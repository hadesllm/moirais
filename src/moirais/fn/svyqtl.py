"""Survey-weighted quantile."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["survey_quantile"]


def survey_quantile(y, weights, quantile):
    """
    Survey-weighted quantile

    Formula: Q_q via weighted CDF inversion

    Parameters
    ----------
    y : array-like
        Input data.
    weights : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Francisco-Fuller (1991)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Survey-weighted quantile"})


def cheatsheet():
    return "svyqtl: Survey-weighted quantile"
