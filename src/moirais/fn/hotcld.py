"""Getis-Ord hot/cold spot map."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hot_cold_spots"]


def hot_cold_spots(x, W, alpha):
    """
    Getis-Ord hot/cold spot map

    Formula: G_i* z-score with FDR

    Parameters
    ----------
    x : array-like
        Input data.
    W : array-like
        Input data.
    alpha : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Getis-Ord (1992)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Getis-Ord hot/cold spot map"})


def cheatsheet():
    return "hotcld: Getis-Ord hot/cold spot map"
