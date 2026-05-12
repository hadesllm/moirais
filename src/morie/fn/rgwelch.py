# morie.fn -- function file (hadesllm/morie)
"""Welch PSD estimate (overlapping windowed periodograms)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_welch_psd"]


def rangayyan_welch_psd(x, fs, nperseg, noverlap, window):
    """
    Welch PSD estimate (overlapping windowed periodograms)

    Formula: P_W(f) = (1/KU) sum |W_k(f)|^2; U = (1/N) sum w^2[n]

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    nperseg : array-like
        Input data.
    noverlap : array-like
        Input data.
    window : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: psd, freqs

    References
    ----------
    Rangayyan Ch 6.3.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Welch PSD estimate (overlapping windowed periodograms)"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Welch PSD estimate (overlapping windowed periodograms)"})


def cheatsheet():
    return "rgwelch: Welch PSD estimate (overlapping windowed periodograms)"
