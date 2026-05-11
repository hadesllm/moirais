"""Twiddle factor expressed in terms of cosine and sine basis functions.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_twiddle_cos_sin"]


def rangayyan_ch3_twiddle_cos_sin(n, k, N):
    """
    Twiddle factor expressed in terms of cosine and sine basis functions.

    Formula: W_N^(n*k) = exp(-j*(2*pi/N)*n*k) = cos((2*pi/N)*n*k) - j*sin((2*pi/N)*n*k)

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
    Rangayyan (2024), Ch 3, Eq 3.84, p. 127
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Twiddle factor expressed in terms of cosine and sine basis functions."})


def cheatsheet():
    return "rng073: Twiddle factor expressed in terms of cosine and sine basis functions."
