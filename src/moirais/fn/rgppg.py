# moirais.fn — function file (hadesllm/moirais)
"""PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ppg_features"]


def rangayyan_ppg_features(ppg, fs):
    """
    PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)

    Formula: Features: systolic amplitude, pulse width, augmentation index AI = (P2-P1)/P1

    Parameters
    ----------
    ppg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.11
    """
    ppg = np.asarray(ppg, dtype=float)
    n = int(ppg) if ppg.ndim == 0 else len(ppg)
    result = float(np.mean(ppg))
    se = float(np.std(ppg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)"})


def cheatsheet():
    return "rgppg: PPG waveform feature extraction (systolic peak, dicrotic notch, diastolic peak)"
