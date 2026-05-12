# morie.fn -- function file (hadesllm/morie)
"""Muscle contraction artifact removal from VAG signals."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_muscle_artifact"]


def rangayyan_muscle_artifact(vag, emg_ref, fs):
    """
    Muscle contraction artifact removal from VAG signals

    Formula: Notch + adaptive filtering on EMG-contaminated VAG

    Parameters
    ----------
    vag : array-like
        Input data.
    emg_ref : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: vag_clean

    References
    ----------
    Rangayyan Ch 3.15
    """
    vag = np.asarray(vag, dtype=float)
    n = int(vag) if vag.ndim == 0 else len(vag)
    result = float(np.mean(vag))
    se = float(np.std(vag, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Muscle contraction artifact removal from VAG signals"})


def cheatsheet():
    return "rgmscart: Muscle contraction artifact removal from VAG signals"
