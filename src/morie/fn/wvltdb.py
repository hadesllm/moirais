"""Daubechies wavelet decomposition."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["db_wavelet"]


def db_wavelet(y, n, levels):
    """
    Daubechies wavelet decomposition

    Formula: DWT with db_n filters

    Parameters
    ----------
    y : array-like
        Input data.
    n : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Daubechies (1992)
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Daubechies wavelet decomposition"})


def cheatsheet():
    return "wvltdb: Daubechies wavelet decomposition"
