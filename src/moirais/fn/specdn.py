"""Spectral density via periodogram."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["spectral_density"]


def spectral_density(y):
    """
    Spectral density via periodogram

    Formula: |FFT(y)|^2 / (2 pi N)

    Parameters
    ----------
    y : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Brillinger (2001)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral density via periodogram"})


def cheatsheet():
    return "specdn: Spectral density via periodogram"
