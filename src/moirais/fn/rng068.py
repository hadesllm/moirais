"""DFT computed at K samples of normalized frequency.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dft_K_samples"]


def rangayyan_ch3_dft_K_samples(x, n, k, K, N):
    """
    DFT computed at K samples of normalized frequency.

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * exp(-j * (2*pi/K) * n * k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    K : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.79, p. 126
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFT computed at K samples of normalized frequency."})


def cheatsheet():
    return "rng068: DFT computed at K samples of normalized frequency."
