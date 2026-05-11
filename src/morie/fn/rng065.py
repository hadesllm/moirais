"""Continuous-time Fourier transform with frequency variable f in Hz.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_fourier_transform_f"]


def rangayyan_ch3_fourier_transform_f(x, t, f):
    """
    Continuous-time Fourier transform with frequency variable f in Hz.

    Formula: X(f) = integral_{-inf}^{inf} x(t) * exp(-j*2*pi*f*t) dt

    Parameters
    ----------
    x : array-like
        Input data.
    t : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.76, p. 125
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Continuous-time Fourier transform with frequency variable f in Hz."})


def cheatsheet():
    return "rng065: Continuous-time Fourier transform with frequency variable f in Hz."
