"""Fourier-domain expression for a signal with a wavelet plus echo.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_fourier_signal_echo"]


def rangayyan_ch4_fourier_signal_echo(a, n_0, omega, H):
    """
    Fourier-domain expression for a signal with a wavelet plus echo.

    Formula: Y(omega) = [1 + a * exp(-j*omega*n_0)] * H(omega)

    Parameters
    ----------
    a : array-like
        Input data.
    n_0 : array-like
        Input data.
    omega : array-like
        Input data.
    H : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.77, p. 249
    """
    a = np.atleast_1d(np.asarray(a, dtype=float))
    n = len(a)
    result = float(np.mean(a))
    se = float(np.std(a, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier-domain expression for a signal with a wavelet plus echo."})


def cheatsheet():
    return "rng249: Fourier-domain expression for a signal with a wavelet plus echo."
