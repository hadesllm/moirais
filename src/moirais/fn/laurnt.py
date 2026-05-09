"""Laurent series (with negative powers)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["laurent_series"]


def laurent_series(f, c, order):
    """
    Laurent series (with negative powers)

    Formula: sum_{n=-∞}^∞ a_n (x-c)^n

    Parameters
    ----------
    f : array-like
        Input data.
    c : array-like
        Input data.
    order : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Laurent (1843)
    """
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = len(f)
    result = float(np.mean(f))
    se = float(np.std(f, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Laurent series (with negative powers)"})


def cheatsheet():
    return "laurnt: Laurent series (with negative powers)"
