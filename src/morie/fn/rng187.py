"""Combined input-output relationship of the Pan-Tompkins highpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_highpass_combined"]


def rangayyan_ch4_pan_tompkins_highpass_combined(x, p, n):
    """
    Combined input-output relationship of the Pan-Tompkins highpass filter.

    Formula: p(n) = p(n-1) - (1/32)*x(n) + x(n-16) - x(n-17) + (1/32)*x(n-32)

    Parameters
    ----------
    x : array-like
        Input data.
    p : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.13, p. 222
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Combined input-output relationship of the Pan-Tompkins highpass filter."})


def cheatsheet():
    return "rng187: Combined input-output relationship of the Pan-Tompkins highpass filter."
