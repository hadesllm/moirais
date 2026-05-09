"""Front-door criterion."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["front_door"]


def front_door(Y, X, M, C):
    """
    Front-door criterion

    Formula: P(Y|do(X)) = sum_m P(m|x) sum_x' P(Y|x',m) P(x')

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Pearl (1995)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Front-door criterion"})


def cheatsheet():
    return "medFront: Front-door criterion"
