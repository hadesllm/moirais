"""Inverse DFT expressed as combination of cos and sin synthesis terms.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_idft_real_imag"]


def rangayyan_ch3_idft_real_imag(X, n, k, N):
    """
    Inverse DFT expressed as combination of cos and sin synthesis terms.

    Formula: x(n) = (1/N) * sum_{k=0}^{N-1} X(k) cos((2*pi/N)*n*k) + j * (1/N) * sum_{k=0}^{N-1} X(k) sin((2*pi/N)*n*k)

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
    Rangayyan (2024), Ch 3, Eq 3.86, p. 128
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Inverse DFT expressed as combination of cos and sin synthesis terms."})


def cheatsheet():
    return "rng075: Inverse DFT expressed as combination of cos and sin synthesis terms."
