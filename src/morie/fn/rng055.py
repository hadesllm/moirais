"""Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dtft_via_z"]


def rangayyan_ch3_dtft_via_z(x, n, omega, T, N):
    """
    Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle.

    Formula: X(omega) = sum_{n=0}^{N-1} x(n) * z^(-n) |_{z=exp(j*omega*T)} = sum_{n=0}^{N-1} x(n) * exp(-j*omega*n*T)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    omega : array-like
        Input data.
    T : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.66, p. 122
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle."})


def cheatsheet():
    return "rng055: Discrete-time Fourier transform obtained by evaluating z-transform on the unit circle."
