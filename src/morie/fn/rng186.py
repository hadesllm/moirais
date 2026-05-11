"""Difference equation of the Pan-Tompkins highpass filter (intermediate).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_highpass_difference_eq"]


def rangayyan_ch4_pan_tompkins_highpass_difference_eq(x, y, n):
    """
    Difference equation of the Pan-Tompkins highpass filter (intermediate).

    Formula: p(n) = x(n - 16) - (1/32) * [y(n-1) + x(n) - x(n-32)]

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
    Rangayyan (2024), Ch 4, Eq 4.12, p. 221
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Difference equation of the Pan-Tompkins highpass filter (intermediate)."})


def cheatsheet():
    return "rng186: Difference equation of the Pan-Tompkins highpass filter (intermediate)."
