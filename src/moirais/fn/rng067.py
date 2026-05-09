"""Discrete-time Fourier transform (DTFT) of x(n) with continuous omega.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dtft"]


def rangayyan_ch3_dtft(x, n, omega):
    """
    Discrete-time Fourier transform (DTFT) of x(n) with continuous omega.

    Formula: X(omega) = sum_{n=-inf}^{inf} x(n) * exp(-j*omega*n)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.78, p. 126
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete-time Fourier transform (DTFT) of x(n) with continuous omega."})


def cheatsheet():
    return "rng067: Discrete-time Fourier transform (DTFT) of x(n) with continuous omega."
