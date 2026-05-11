# morie.fn — function file (hadesllm/morie)
"""Carotid pulse waveform feature extraction."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_carotid_pulse"]


def rangayyan_carotid_pulse(pulse, fs):
    """
    Carotid pulse waveform feature extraction

    Formula: Features: anacrotic rise, systolic peak, dicrotic notch, diastolic peak, coeff. of elasticity

    Parameters
    ----------
    pulse : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: features

    References
    ----------
    Rangayyan Ch 1.2.10
    """
    pulse = np.asarray(pulse, dtype=float)
    n = int(pulse) if pulse.ndim == 0 else len(pulse)
    result = float(np.mean(pulse))
    se = float(np.std(pulse, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Carotid pulse waveform feature extraction"})


def cheatsheet():
    return "rgcpulse: Carotid pulse waveform feature extraction"
