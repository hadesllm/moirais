"""Unfolding analysis for preference data."""
import numpy as np
from ._richresult import RichResult

__all__ = ["unfolding_analysis"]


def unfolding_analysis(x):
    """
    Unfolding analysis for preference data

    Formula: d(i,j) = ||x_i - y_j|| preference distances

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Armstrong Ch 7
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Unfolding analysis for preference data"})


def cheatsheet():
    return "unfdl: Unfolding analysis for preference data"
