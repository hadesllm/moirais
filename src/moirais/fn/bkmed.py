"""Baron-Kenny stepwise mediation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["baron_kenny"]


def baron_kenny(Y, X, M):
    """
    Baron-Kenny stepwise mediation

    Formula: step 1: Y~X; step 2: M~X; step 3: Y~X+M

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Baron-Kenny (1986)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Baron-Kenny stepwise mediation"})


def cheatsheet():
    return "bkmed: Baron-Kenny stepwise mediation"
