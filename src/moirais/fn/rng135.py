"""Butterworth highpass response indexed by DFT bin k.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_highpass_dft_indexed"]


def rangayyan_ch3_butterworth_highpass_dft_indexed(k, k_c, N):
    """
    Butterworth highpass response indexed by DFT bin k.

    Formula: |H(k)|^2 = 1 / (1 + (k_c/k)^(2*N))

    Parameters
    ----------
    k : array-like
        Input data.
    k_c : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.149, p. 161
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    n = len(k)
    result = float(np.mean(k))
    se = float(np.std(k, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Butterworth highpass response indexed by DFT bin k."})


def cheatsheet():
    return "rng135: Butterworth highpass response indexed by DFT bin k."
