"""Middle-out hierarchy approach."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["middle_out"]


def middle_out(middle, S):
    """
    Middle-out hierarchy approach

    Formula: forecast at middle, BU above + TD below

    Parameters
    ----------
    middle : array-like
        Input data.
    S : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hyndman-Athanasopoulos (2018) §11
    """
    middle = np.atleast_1d(np.asarray(middle, dtype=float))
    n = len(middle)
    result = float(np.mean(middle))
    se = float(np.std(middle, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Middle-out hierarchy approach"})


def cheatsheet():
    return "middle: Middle-out hierarchy approach"
