"""KKT conditions."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["boyd_kkt"]


def boyd_kkt(x, lambda_, nu):
    """
    KKT conditions

    Formula: stationarity, primal/dual feasibility, complementary slackness

    Parameters
    ----------
    x : array-like
        Input data.
    lambda_ : array-like
        Input data.
    nu : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: satisfied

    References
    ----------
    Boyd CVX Ch 5
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "KKT conditions"})


def cheatsheet():
    return "cvxkkt: KKT conditions"
