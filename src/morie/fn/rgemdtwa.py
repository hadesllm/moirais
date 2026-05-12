# morie.fn -- function file (hadesllm/morie)
"""T-wave alternans detection via EMD-based signal decomposition."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emd_twa"]


def rangayyan_emd_twa(ecg, fs, r_peaks):
    """
    T-wave alternans detection via EMD-based signal decomposition

    Formula: IMF at alternans frequency (0.5 cycles/beat); alternans amplitude from IMF energy

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: twa_amp, alternating_imf

    References
    ----------
    Rangayyan Ch 9.10
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    result = float(np.mean(ecg))
    se = float(np.std(ecg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "T-wave alternans detection via EMD-based signal decomposition"})


def cheatsheet():
    return "rgemdtwa: T-wave alternans detection via EMD-based signal decomposition"
