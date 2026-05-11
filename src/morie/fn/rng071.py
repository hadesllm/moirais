"""Twiddle factor used in DFT and FFT formulations.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_twiddle_factor"]


def rangayyan_ch3_twiddle_factor(N):
    """
    Twiddle factor used in DFT and FFT formulations.

    Formula: W_N = exp(-j * 2*pi / N)

    Parameters
    ----------
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.82, p. 127
    """
    N = np.atleast_1d(np.asarray(N, dtype=float))
    n = len(N)
    result = float(np.mean(N))
    se = float(np.std(N, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Twiddle factor used in DFT and FFT formulations."})


def cheatsheet():
    return "rng071: Twiddle factor used in DFT and FFT formulations."
