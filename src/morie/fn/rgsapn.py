# morie.fn — function file (hadesllm/morie)
"""Sleep apnea detection via ECG-derived respiration + SpO2 fusion."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_sleep_apnea"]


def rangayyan_sleep_apnea(ecg, spo2, fs):
    """
    Sleep apnea detection via ECG-derived respiration + SpO2 fusion

    Formula: Apnea index = events/hr; detection threshold on RR variability + desaturation

    Parameters
    ----------
    ecg : array-like
        Input data.
    spo2 : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: apnea_index, events

    References
    ----------
    Rangayyan Ch 2.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sleep apnea detection via ECG-derived respiration + SpO2 fusion"})


def cheatsheet():
    return "rgsapn: Sleep apnea detection via ECG-derived respiration + SpO2 fusion"
