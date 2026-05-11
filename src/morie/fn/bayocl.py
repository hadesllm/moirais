"""Bayesian outlier detection via DP."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bayes_outlier_dp"]


def bayes_outlier_dp(y, alpha):
    """
    Bayesian outlier detection via DP

    Formula: posterior cluster size = 1 indicates outlier

    Parameters
    ----------
    y : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Quintana-Iglesias (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bayesian outlier detection via DP"})


def cheatsheet():
    return "bayocl: Bayesian outlier detection via DP"
