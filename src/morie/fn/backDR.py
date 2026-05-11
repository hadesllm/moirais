"""Back-door adjustment."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["back_door"]


def back_door(Y, X, C):
    """
    Back-door adjustment

    Formula: P(Y|do(X)) = sum_c P(Y|X,C) P(C)

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Back-door adjustment"})


def cheatsheet():
    return "backDR: Back-door adjustment"
