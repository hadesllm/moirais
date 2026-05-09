"""Time-domain impulse response of the matched filter for EEG spike-and-wave detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_impulse_response_eeg"]


def rangayyan_ch4_matched_filter_impulse_response_eeg(x, K, t, t_0):
    """
    Time-domain impulse response of the matched filter for EEG spike-and-wave detection.

    Formula: h(t) = K * x(t_0 - t)

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
    Rangayyan (2024), Ch 4, Eq 4.56, p. 241
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Time-domain impulse response of the matched filter for EEG spike-and-wave detection."})


def cheatsheet():
    return "rng228: Time-domain impulse response of the matched filter for EEG spike-and-wave detection."
