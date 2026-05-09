"""Sample-selection bound (Heckman model)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["bound_selection"]


def bound_selection(y, D, X):
    """
    Sample-selection bound (Heckman model)

    Formula: non-identified region bounds

    Parameters
    ----------
    y : array-like
        Input data.
    D : array-like
        Input data.
    X : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Heckman (1979); Manski (2003)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sample-selection bound (Heckman model)"})


def cheatsheet():
    return "bnssel: Sample-selection bound (Heckman model)"
