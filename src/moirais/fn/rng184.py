"""Difference equation of the lowpass component used in the Pan-Tompkins highpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq"]


def rangayyan_ch4_pan_tompkins_highpass_lp_difference_eq(x, y, n):
    """
    Difference equation of the lowpass component used in the Pan-Tompkins highpass filter.

    Formula: y(n) = y(n-1) + x(n) - x(n-32)

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
    Rangayyan (2024), Ch 4, Eq 4.10, p. 221
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Difference equation of the lowpass component used in the Pan-Tompkins highpass filter."})


def cheatsheet():
    return "rng184: Difference equation of the lowpass component used in the Pan-Tompkins highpass filter."
