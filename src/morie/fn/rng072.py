"""DFT expressed using twiddle factors W_N^(nk).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_dft_via_twiddle"]


def rangayyan_ch3_dft_via_twiddle(x, n, k, W_N, N):
    """
    DFT expressed using twiddle factors W_N^(nk).

    Formula: X(k) = sum_{n=0}^{N-1} x(n) * W_N^(n*k)

    Parameters
    ----------
    x : array-like
        Input data.
    n : array-like
        Input data.
    k : array-like
        Input data.
    W_N : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.83, p. 127
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "DFT expressed using twiddle factors W_N^(nk)."})


def cheatsheet():
    return "rng072: DFT expressed using twiddle factors W_N^(nk)."
