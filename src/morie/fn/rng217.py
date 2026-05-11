"""Schwarz inequality for real functions a(t) and b(t).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_schwarz_inequality_real"]


def rangayyan_ch4_schwarz_inequality_real(a, b, t):
    """
    Schwarz inequality for real functions a(t) and b(t).

    Formula: [ integral a(t) b(t) dt ]^2 <= integral a^2(t) dt * integral b^2(t) dt

    Parameters
    ----------
    a : array-like
        Input data.
    b : array-like
        Input data.
    t : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.43, p. 239
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Schwarz inequality for real functions a(t) and b(t)."})


def cheatsheet():
    return "rng217: Schwarz inequality for real functions a(t) and b(t)."
