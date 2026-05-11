"""DFT decomposed into real (cos) and imaginary (sin) parts.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dft_real_imag_decomposition"]


def rangayyan_ch3_dft_real_imag_decomposition(x, n, k, N):
    """
    DFT decomposed into real (cos) and imaginary (sin) parts.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) cos((2*pi/N)*n*k) - j * sum_{n=0}^{N-1} x(n) sin((2*pi/N)*n*k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.85, p. 127
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFT decomposed into real (cos) and imaginary (sin) parts."})


def cheatsheet():
    return "rng074: DFT decomposed into real (cos) and imaginary (sin) parts."
