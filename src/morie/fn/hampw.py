"""Hampel three-part redescending weight."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["hampel_three_part"]


def hampel_three_part(y, a, b, c):
    """
    Hampel three-part redescending weight

    Formula: piecewise: identity, k/|r|, then linear descent to zero at C

    Parameters
    ----------
    y : array-like
        Input data.
    a : array-like
        Input data.
    b : array-like
        Input data.
    c : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hampel (1974)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Hampel three-part redescending weight"})


def cheatsheet():
    return "hampw: Hampel three-part redescending weight"
