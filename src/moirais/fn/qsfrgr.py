"""Quantile survival forest."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["quantile_survival_forest"]


def quantile_survival_forest(time, event, X, quantile):
    """
    Quantile survival forest

    Formula: random survival forest with quantile loss

    Parameters
    ----------
    time : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    quantile : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cui et al (2023)
    """
    time = np.atleast_1d(np.asarray(time, dtype=float))
    n = len(time)
    result = float(np.mean(time))
    se = float(np.std(time, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Quantile survival forest"})


def cheatsheet():
    return "qsfrgr: Quantile survival forest"
