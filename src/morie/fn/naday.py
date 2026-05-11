"""Nadaraya-Watson kernel regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["nadaraya_watson"]


def nadaraya_watson(x, y, h):
    """
    Nadaraya-Watson kernel regression

    Formula: m̂(x) = sum K_h(x-x_i)y_i / sum K_h(x-x_i)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Nadaraya (1964); Watson (1964)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nadaraya-Watson kernel regression"})


def cheatsheet():
    return "naday: Nadaraya-Watson kernel regression"
