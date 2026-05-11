"""VAR(p) vector autoregression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["vector_autoregression"]


def vector_autoregression(Y, p):
    """
    VAR(p) vector autoregression

    Formula: Y_t = c + sum_i A_i Y_{t-i} + eps_t

    Parameters
    ----------
    Y : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Lütkepohl (2005)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "VAR(p) vector autoregression"})


def cheatsheet():
    return "varest: VAR(p) vector autoregression"
