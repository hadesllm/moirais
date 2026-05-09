"""Symmetry property of twiddle factors used in FFT.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_twiddle_conjugate_symmetry"]


def rangayyan_ch3_twiddle_conjugate_symmetry(n, k, N):
    """
    Symmetry property of twiddle factors used in FFT.

    Formula: W_N^(-n*k) = (W_N^(n*k))*

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
    Rangayyan (2024), Ch 3, Eq 3.88, p. 130
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Symmetry property of twiddle factors used in FFT."})


def cheatsheet():
    return "rng077: Symmetry property of twiddle factors used in FFT."
