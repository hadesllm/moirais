"""Impulse response of the Hann smoothing filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_hann_impulse_response"]


def rangayyan_ch3_hann_impulse_response(n):
    """
    Impulse response of the Hann smoothing filter.

    Formula: h(n) = (1/4) * [delta(n) + 2*delta(n-1) + delta(n-2)]

    Parameters
    ----------
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.101, p. 140
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Impulse response of the Hann smoothing filter."})


def cheatsheet():
    return "rng090: Impulse response of the Hann smoothing filter."
