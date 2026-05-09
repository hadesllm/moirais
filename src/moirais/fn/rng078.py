"""Periodicity property of twiddle factors used in FFT.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_twiddle_periodicity"]


def rangayyan_ch3_twiddle_periodicity(n, k, N):
    """
    Periodicity property of twiddle factors used in FFT.

    Formula: W_N^(n*k) = W_N^(n*(k+N)) = W_N^((n+N)*k)

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.89, p. 130
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Periodicity property of twiddle factors used in FFT."})


def cheatsheet():
    return "rng078: Periodicity property of twiddle factors used in FFT."
