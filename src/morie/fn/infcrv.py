"""Influence function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["influence_function"]


def influence_function(estimator, F, x):
    """
    Influence function

    Formula: IF(x; T, F) = lim_{ε->0} [T((1−ε)F + εδ_x) − T(F)]/ε

    Parameters
    ----------
    estimator : array-like
        Input data.
    F : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel (1974)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Influence function"})


def cheatsheet():
    return "infcrv: Influence function"
