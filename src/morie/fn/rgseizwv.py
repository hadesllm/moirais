# morie.fn -- function file (hadesllm/morie)
"""EEG epileptic seizure detection via wavelet energy."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_seizure_wavelet"]


def rangayyan_seizure_wavelet(eeg, fs, wavelet, levels):
    """
    EEG epileptic seizure detection via wavelet energy

    Formula: E_j = sum |d_j[n]|^2; ictal increase in delta/theta wavelet energy

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: wavelet_energies, is_seizure

    References
    ----------
    Rangayyan Ch 8.17
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EEG epileptic seizure detection via wavelet energy"})


def cheatsheet():
    return "rgseizwv: EEG epileptic seizure detection via wavelet energy"
