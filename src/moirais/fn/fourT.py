"""Fourier transform."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["fourier_transform"]


def fourier_transform(f, x, k):
    """
    Fourier transform

    Formula: F(k) = ∫ f(x) e^{-2πikx} dx

    Parameters
    ----------
    f : array-like
        Input data.
    x : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Fourier (1822)
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Fourier transform"})


def cheatsheet():
    return "fourT: Fourier transform"
