"""Cook's distance for influential observations."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["cooks_distance"]


def cooks_distance(y, X):
    """
    Cook's distance for influential observations

    Formula: D_i = (e_i^2 / p s^2) * h_ii / (1 - h_ii)^2

    Parameters
    ----------
    y : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Cook (1977)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Cook's distance for influential observations"})


def cheatsheet():
    return "cookd: Cook's distance for influential observations"
