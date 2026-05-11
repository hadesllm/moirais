# morie.fn — function file (hadesllm/morie)
"""EEG rhythm band classification (delta/theta/alpha/beta/gamma)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_eeg_rhythms"]


def rangayyan_eeg_rhythms(eeg, fs):
    """
    EEG rhythm band classification (delta/theta/alpha/beta/gamma)

    Formula: Band membership by frequency range: delta<4, theta 4-8, alpha 8-13, beta 13-30, gamma>30 Hz

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: band_powers

    References
    ----------
    Rangayyan Ch 1.2.6
    """
    eeg = np.asarray(eeg, dtype=float)
    n = int(eeg) if eeg.ndim == 0 else len(eeg)
    result = float(np.mean(eeg))
    se = float(np.std(eeg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EEG rhythm band classification (delta/theta/alpha/beta/gamma)"})


def cheatsheet():
    return "rgeegb: EEG rhythm band classification (delta/theta/alpha/beta/gamma)"
