"""Expected value E[X]."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_expectation"]


def wasserman_expectation(x, f):
    """
    Expected value E[X]

    Formula: E[X] = int x f(x) dx

    Parameters
    ----------
    x : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Wasserman (2004), Ch 3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Expected value E[X]"})


def cheatsheet():
    return "wsmexp: Expected value E[X]"
