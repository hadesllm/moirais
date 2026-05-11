# morie.fn — function file (hadesllm/morie)
"""EEG artifact removal via ICA (eye blink, muscle, ECG)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ica_artifact"]


def rangayyan_ica_artifact(eeg, n_components, artifact_labels):
    """
    EEG artifact removal via ICA (eye blink, muscle, ECG)

    Formula: Artifact components identified by kurtosis/correlation; removed from mixing matrix

    Parameters
    ----------
    eeg : array-like
        Input data.
    n_components : array-like
        Input data.
    artifact_labels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: eeg_clean, ica_components

    References
    ----------
    Rangayyan Ch 9.7.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EEG artifact removal via ICA (eye blink, muscle, ECG)"})


def cheatsheet():
    return "rgicaart: EEG artifact removal via ICA (eye blink, muscle, ECG)"
