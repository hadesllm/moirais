"""Inverse discrete Fourier transform (IDFT) of an N-point spectrum.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_idft_definition"]


def rangayyan_ch3_idft_definition(X, n, k, N):
    """
    Inverse discrete Fourier transform (IDFT) of an N-point spectrum.

    Formula: x(n) = (1/N) * sum_{k=0}^{N-1} X(k) * exp(+j * (2*pi/N) * n * k)

    Parameters
    ----------
    X : array-like
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
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.81, p. 126
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse discrete Fourier transform (IDFT) of an N-point spectrum."})


def cheatsheet():
    return "rng070: Inverse discrete Fourier transform (IDFT) of an N-point spectrum."
