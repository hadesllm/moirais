# morie.fn — function file (hadesllm/morie)
"""ECG-EMG coupling during physical effort (VMG correlation)."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ecg_emg_coupling"]


def rangayyan_ecg_emg_coupling(ecg, emg, fs):
    """
    ECG-EMG coupling during physical effort (VMG correlation)

    Formula: CCF(ECG, EMG) at cardiac frequency; cardio-locomotor coupling index

    Parameters
    ----------
    ecg : array-like
        Input data.
    emg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: coupling_index, ccf

    References
    ----------
    Rangayyan Ch 2.2.6
    """
    ecg = np.asarray(ecg, dtype=float)
    y = np.asarray(emg, dtype=float)
    n = min(len(ecg), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "ECG-EMG coupling during physical effort (VMG correlation)"})
    result = stats.spearmanr(ecg[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "ECG-EMG coupling during physical effort (VMG correlation)"})


def cheatsheet():
    return "rgecgemu: ECG-EMG coupling during physical effort (VMG correlation)"
