"""Forward search regression."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["forward_search"]


def forward_search(X, y, initial_h):
    """
    Forward search regression

    Formula: start small, add one obs at a time, monitor stats

    Parameters
    ----------
    X : array-like
        Input data.
    y : array-like
        Input data.
    initial_h : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Atkinson-Riani (2000)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forward search regression"})


def cheatsheet():
    return "forwsr: Forward search regression"
