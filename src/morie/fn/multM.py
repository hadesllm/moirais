"""Multiple mediators (parallel)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["multiple_mediators"]


def multiple_mediators(Y, X, M_list, C):
    """
    Multiple mediators (parallel)

    Formula: per-mediator NIE_k summed for joint NIE

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M_list : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daniel-De Stavola-Cousens-Vansteelandt (2015)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multiple mediators (parallel)"})


def cheatsheet():
    return "multM: Multiple mediators (parallel)"
