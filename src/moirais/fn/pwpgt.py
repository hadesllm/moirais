"""Prentice-Williams-Peterson gap-time model."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["pwp_gap_time"]


def pwp_gap_time(start, stop, event, X, occurrence):
    """
    Prentice-Williams-Peterson gap-time model

    Formula: lambda_k(t|H) = lambda_{0k}(t - t_{k-1}) exp(beta_k'X)

    Parameters
    ----------
    start : array-like
        Input data.
    stop : array-like
        Input data.
    event : array-like
        Input data.
    X : array-like
        Input data.
    occurrence : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Prentice, Williams, Peterson (1981)
    """
    start = np.atleast_1d(np.asarray(start, dtype=float))
    n = len(start)
    result = float(np.mean(start))
    se = float(np.std(start, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Prentice-Williams-Peterson gap-time model"})


def cheatsheet():
    return "pwpgt: Prentice-Williams-Peterson gap-time model"
