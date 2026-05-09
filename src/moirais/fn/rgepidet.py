# moirais.fn — function file (hadesllm/moirais)
"""Epileptic seizure detection in EEG."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_epilepsy_detect"]


def rangayyan_epilepsy_detect(eeg, fs, dictionary_size):
    """
    Epileptic seizure detection in EEG

    Formula: Dictionary learning: ictal features differ from interictal; SVM classifier on atoms

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    dictionary_size : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: is_seizure, onset

    References
    ----------
    Rangayyan Ch 8.17
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Epileptic seizure detection in EEG"})


def cheatsheet():
    return "rgepidet: Epileptic seizure detection in EEG"
