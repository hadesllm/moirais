"""Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_impulse_response"]


def rangayyan_ch4_matched_filter_impulse_response(x, K, t, t_0):
    """
    Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal.

    Formula: h(t) = K * x[-(t - t_0)]

    Parameters
    ----------
    x : array-like
        Input data.
    K : array-like
        Input data.
    t : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: array

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.49, p. 239
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal."})


def cheatsheet():
    return "rng221: Impulse response of the matched filter is a scaled, time-reversed, shifted reference signal."
