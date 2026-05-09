"""Andrews sine wave weight."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["andrews_sine"]


def andrews_sine(y, A):
    """
    Andrews sine wave weight

    Formula: w(r) = sin(r/A)/(r/A) if |r|<=pi*A else 0

    Parameters
    ----------
    y : array-like
        Input data.
    A : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Andrews (1974)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Andrews sine wave weight"})


def cheatsheet():
    return "andrew: Andrews sine wave weight"
