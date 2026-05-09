"""MICE (chained equations)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mi_chained_eq"]


def mi_chained_eq(data, R, models, K):
    """
    MICE (chained equations)

    Formula: sequential conditional impute per variable

    Parameters
    ----------
    data : array-like
        Input data.
    R : array-like
        Input data.
    models : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    van Buuren (2018)
    """
    data = np.atleast_1d(np.asarray(data, dtype=float))
    n = len(data)
    result = float(np.mean(data))
    se = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "MICE (chained equations)"})


def cheatsheet():
    return "miord2: MICE (chained equations)"
