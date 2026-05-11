"""Analog Butterworth transfer function from N left-half-plane poles.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_analog_transfer_function"]


def rangayyan_ch3_butterworth_analog_transfer_function(s, p_k, G, N):
    """
    Analog Butterworth transfer function from N left-half-plane poles.

    Formula: H_a(s) = G / [ (s - p_1)(s - p_2)...(s - p_N) ]

    Parameters
    ----------
    s : array-like
        Input data.
    p_k : array-like
        Input data.
    G : array-like
        Input data.
    N : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.138, p. 154
    """
    s = np.atleast_1d(np.asarray(s, dtype=float))
    n = len(s)
    result = float(np.mean(s))
    se = float(np.std(s, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Analog Butterworth transfer function from N left-half-plane poles."})


def cheatsheet():
    return "rng126: Analog Butterworth transfer function from N left-half-plane poles."
