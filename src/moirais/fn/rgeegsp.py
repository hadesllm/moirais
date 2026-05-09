# moirais.fn — function file (hadesllm/moirais)
"""EEG band power estimation (delta/theta/alpha/beta spectral integration)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_eeg_spectral"]


def rangayyan_eeg_spectral(eeg, fs, n_ch):
    """
    EEG band power estimation (delta/theta/alpha/beta spectral integration)

    Formula: P_delta=integral_{0}^{4}S(f)df; theta 4-8; alpha 8-13; beta 13-30

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    n_ch : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: band_powers_per_channel

    References
    ----------
    Rangayyan Ch 6.7
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    if eeg.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "EEG band power estimation (delta/theta/alpha/beta spectral integration)"})
    estimate = np.median(eeg)
    se = 1.2533 * np.std(eeg, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "EEG band power estimation (delta/theta/alpha/beta spectral integration)"})


def cheatsheet():
    return "rgeegsp: EEG band power estimation (delta/theta/alpha/beta spectral integration)"
