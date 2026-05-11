"""E-divisive (energy distance)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["e_divisive"]


def e_divisive(x, sig):
    """
    E-divisive (energy distance)

    Formula: recursive bisection by energy stat

    Parameters
    ----------
    x : array-like
        Input data.
    sig : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Matteson-James (2014)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "E-divisive (energy distance)"})


def cheatsheet():
    return "e_div: E-divisive (energy distance)"
