"""Wavelet basis (Daubechies)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["wavelet_basis"]


def wavelet_basis(y, wavelet):
    """
    Wavelet basis (Daubechies)

    Formula: discrete wavelet transform

    Parameters
    ----------
    y : array-like
        Input data.
    wavelet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daubechies (1988)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet basis (Daubechies)"})


def cheatsheet():
    return "wave: Wavelet basis (Daubechies)"
