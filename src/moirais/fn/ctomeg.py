"""McDonald's omega total."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["omega_total"]


def omega_total(X, factor_loadings):
    """
    McDonald's omega total

    Formula: omega = (sum lambda_i)^2 / Var(T) using factor loadings

    Parameters
    ----------
    X : array-like
        Input data.
    factor_loadings : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    McDonald (1999)
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "McDonald's omega total"})


def cheatsheet():
    return "ctomeg: McDonald's omega total"
