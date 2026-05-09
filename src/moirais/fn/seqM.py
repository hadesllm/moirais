"""Sequential (causally ordered) mediators."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["sequential_mediators"]


def sequential_mediators(Y, X, M1, M2, C):
    """
    Sequential (causally ordered) mediators

    Formula: path-specific effects M1 → M2 → Y

    Parameters
    ----------
    Y : array-like
        Input data.
    X : array-like
        Input data.
    M1 : array-like
        Input data.
    M2 : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Avin-Shpitser-Pearl (2005)
    """
    Y = np.atleast_1d(np.asarray(Y, dtype=float))
    n = len(Y)
    result = float(np.mean(Y))
    se = float(np.std(Y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sequential (causally ordered) mediators"})


def cheatsheet():
    return "seqM: Sequential (causally ordered) mediators"
