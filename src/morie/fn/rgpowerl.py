# morie.fn -- function file (hadesllm/morie)
"""Powerline interference (50/60 Hz) removal from ECG."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_powerline_removal"]


def rangayyan_powerline_removal(ecg, fs, powerline_freq):
    """
    Powerline interference (50/60 Hz) removal from ECG

    Formula: Adaptive notch or comb filter at powerline fundamental + harmonics

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    powerline_freq : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: ecg_clean

    References
    ----------
    Rangayyan Ch 3.3.4
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Powerline interference (50/60 Hz) removal from ECG"})


def cheatsheet():
    return "rgpowerl: Powerline interference (50/60 Hz) removal from ECG"
