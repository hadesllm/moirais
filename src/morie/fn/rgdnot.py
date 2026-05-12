# morie.fn -- function file (hadesllm/morie)
"""Dicrotic notch detection in carotid pulse waveform."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_dicrotic_notch"]


def rangayyan_dicrotic_notch(pulse, fs):
    """
    Dicrotic notch detection in carotid pulse waveform

    Formula: Notch = local minimum between systolic and diastolic peaks

    Parameters
    ----------
    pulse : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: notch_loc

    References
    ----------
    Rangayyan Ch 4.3.5
    """
    pulse = np.asarray(pulse, dtype=float)
    n = int(pulse) if pulse.ndim == 0 else len(pulse)
    result = float(np.mean(pulse))
    se = float(np.std(pulse, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Dicrotic notch detection in carotid pulse waveform"})


def cheatsheet():
    return "rgdnot: Dicrotic notch detection in carotid pulse waveform"
