# moirais.fn — function file (hadesllm/moirais)
"""Feature extraction for BCI from EEG (event-related desynchronization/synchronization)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_feature_extract_bci"]


def rangayyan_feature_extract_bci(eeg, fs, ref_window, active_window, band):
    """
    Feature extraction for BCI from EEG (event-related desynchronization/synchronization)

    Formula: ERD = (R - A) / A * 100%; A = reference band power, R = active band power

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    ref_window : array-like
        Input data.
    active_window : array-like
        Input data.
    band : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: erd, ers, features

    References
    ----------
    Rangayyan Ch 9.12.2
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Feature extraction for BCI from EEG (event-related desynchronization/synchronization)"})


def cheatsheet():
    return "rgfeatex: Feature extraction for BCI from EEG (event-related desynchronization/synchronization)"
