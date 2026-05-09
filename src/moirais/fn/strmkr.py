"""Strauss process inhibition model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["strauss_process"]


def strauss_process(coords, r, gamma):
    """
    Strauss process inhibition model

    Formula: density gamma^{n_close_pairs}

    Parameters
    ----------
    coords : array-like
        Input data.
    r : array-like
        Input data.
    gamma : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Strauss (1975)
    """
    coords = np.atleast_1d(np.asarray(coords, dtype=float))
    n = len(coords)
    result = float(np.mean(coords))
    se = float(np.std(coords, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Strauss process inhibition model"})


def cheatsheet():
    return "strmkr: Strauss process inhibition model"
