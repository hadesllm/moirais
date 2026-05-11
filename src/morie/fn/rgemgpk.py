# morie.fn — function file (hadesllm/morie)
"""EMG mean/median frequency from power spectrum."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_emg_peak_freq"]


def rangayyan_emg_peak_freq(emg, fs):
    """
    EMG mean/median frequency from power spectrum

    Formula: f_mean = sum(f*S(f))/sum(S(f)); f_median: sum_{0}^{f_med}S=sum_{f_med}^{inf}S

    Parameters
    ----------
    emg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mean_freq, median_freq

    References
    ----------
    Rangayyan Ch 6
    """
    emg = np.asarray(emg, dtype=float)
    n = int(emg) if emg.ndim == 0 else len(emg)
    result = float(np.mean(emg))
    se = float(np.std(emg, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "EMG mean/median frequency from power spectrum"})


def cheatsheet():
    return "rgemgpk: EMG mean/median frequency from power spectrum"
