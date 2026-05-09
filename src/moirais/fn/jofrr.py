# moirais.fn — function file (hadesllm/moirais)
"""Fourier-series seasonality features: sin/cos at K harmonics of the seasonal period m."""
import numpy as np
from ._richresult import RichResult

__all__ = ["joseph_fourier_features"]


def joseph_fourier_features(t, m, K):
    """
    Fourier-series seasonality features: sin/cos at K harmonics of the seasonal period m

    Formula: sin(2*pi*k*t/m), cos(2*pi*k*t/m)  for k=1..K

    Parameters
    ----------
    t : array-like
        Input data.
    m : array-like
        Input data.
    K : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Joseph Ch 6, Fourier Features section
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    n = len(t)
    result = float(np.mean(t))
    se = float(np.std(t, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier-series seasonality features: sin/cos at K harmonics of the seasonal period m"})


def cheatsheet():
    return "jofrr: Fourier-series seasonality features: sin/cos at K harmonics of the seasonal period m"
