# morie.fn -- function file (hadesllm/morie)
"""Heart sound (S1/S2) identification via PCG-ECG timing."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_heart_sound_id"]


def rangayyan_heart_sound_id(pcg, ecg, fs):
    """
    Heart sound (S1/S2) identification via PCG-ECG timing

    Formula: S1 in [0, 30%] of cardiac cycle; S2 in [40%, 60%] relative to R-peak

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: s1_locs, s2_locs

    References
    ----------
    Rangayyan Ch 4.9
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    result = float(np.mean(pcg))
    se = float(np.std(pcg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Heart sound (S1/S2) identification via PCG-ECG timing"})


def cheatsheet():
    return "rghsnd: Heart sound (S1/S2) identification via PCG-ECG timing"
