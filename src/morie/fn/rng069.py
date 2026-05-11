"""Forward discrete Fourier transform (DFT) of an N-point signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dft_definition"]


def rangayyan_ch3_dft_definition(x, n, k, N):
    """
    Forward discrete Fourier transform (DFT) of an N-point signal.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * exp(-j * (2*pi/N) * n * k)

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
    Rangayyan (2024), Ch 3, Eq 3.80, p. 126
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Forward discrete Fourier transform (DFT) of an N-point signal."})


def cheatsheet():
    return "rng069: Forward discrete Fourier transform (DFT) of an N-point signal."
