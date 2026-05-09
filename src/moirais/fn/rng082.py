"""Odd-symmetric part of a signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_odd_part"]


def rangayyan_ch3_odd_part(x, n):
    """
    Odd-symmetric part of a signal.

    Formula: x_o(n) = 0.5 * [x(n) - x(-n)]

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.93, p. 135
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Odd-symmetric part of a signal."})


def cheatsheet():
    return "rng082: Odd-symmetric part of a signal."
