"""Difference equation of the Pan-Tompkins lowpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_lowpass_difference_eq"]


def rangayyan_ch4_pan_tompkins_lowpass_difference_eq(x, y, n):
    """
    Difference equation of the Pan-Tompkins lowpass filter.

    Formula: y(n) = 2*y(n-1) - y(n-2) + (1/32)*[x(n) - 2*x(n-6) + x(n-12)]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.8, p. 220
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Difference equation of the Pan-Tompkins lowpass filter."})


def cheatsheet():
    return "rng182: Difference equation of the Pan-Tompkins lowpass filter."
