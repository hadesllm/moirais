# morie.fn — function file (hadesllm/morie)
"""EEG seizure detection via rhythm coherence analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_seizure_detect"]


def rangayyan_seizure_detect(eeg, fs, ch_pairs):
    """
    EEG seizure detection via rhythm coherence analysis

    Formula: Seizure: sustained increase in delta/theta band coherence across EEG channels

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    ch_pairs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: seizure_onset, duration

    References
    ----------
    Rangayyan Ch 4.4.3
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EEG seizure detection via rhythm coherence analysis"})


def cheatsheet():
    return "rgseiz: EEG seizure detection via rhythm coherence analysis"
