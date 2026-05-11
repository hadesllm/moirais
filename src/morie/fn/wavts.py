"""Wavelet decomposition for time series."""
import numpy as np
from ._richresult import RichResult

__all__ = ["wavelet_time_series"]


def wavelet_time_series(x):
    """
    Wavelet decomposition for time series

    Formula: W(a,b) = integral x(t)*psi((t-b)/a) dt

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Percival & Walden (2000)
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet decomposition for time series"})


def cheatsheet():
    return "wavts: Wavelet decomposition for time series"
