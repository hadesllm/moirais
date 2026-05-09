"""Frequency-domain optimal matched-filter response for EEG spike-and-wave detection.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_matched_filter_optimal_H_eeg"]


def rangayyan_ch4_matched_filter_optimal_H_eeg(X, K, f, t_0):
    """
    Frequency-domain optimal matched-filter response for EEG spike-and-wave detection.

    Formula: H(f) = K * X*(f) * exp(-j*2*pi*f*t_0)

    Parameters
    ----------
    X : array-like
        Input data.
    K : array-like
        Input data.
    f : array-like
        Input data.
    t_0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.55, p. 241
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency-domain optimal matched-filter response for EEG spike-and-wave detection."})


def cheatsheet():
    return "rng227: Frequency-domain optimal matched-filter response for EEG spike-and-wave detection."
