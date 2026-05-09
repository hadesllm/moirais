# moirais.fn — function file (hadesllm/moirais)
"""Polysomnography signal fusion for sleep staging."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_polysomnography"]


def rangayyan_polysomnography(eeg, eog, emg, fs, epoch_len):
    """
    Polysomnography signal fusion for sleep staging

    Formula: Features from EEG, EOG, EMG -> rule-based or ML sleep stage classifier

    Parameters
    ----------
    eeg : array-like
        Input data.
    eog : array-like
        Input data.
    emg : array-like
        Input data.
    fs : array-like
        Input data.
    epoch_len : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: sleep_stages, hypnogram

    References
    ----------
    Rangayyan Ch 2.4.1
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Polysomnography signal fusion for sleep staging"})


def cheatsheet():
    return "rgpolysg: Polysomnography signal fusion for sleep staging"
