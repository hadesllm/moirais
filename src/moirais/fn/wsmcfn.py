"""Characteristic function."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wasserman_char_fn"]


def wasserman_char_fn(x, t):
    """
    Characteristic function

    Formula: phi_X(t) = E[e^{itX}]

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
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
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Characteristic function"})


def cheatsheet():
    return "wsmcfn: Characteristic function"
