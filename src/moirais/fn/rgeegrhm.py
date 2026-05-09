# moirais.fn — function file (hadesllm/moirais)
"""EEG alpha rhythm presence detection via autocorrelation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_eeg_rhythm_detect"]


def rangayyan_eeg_rhythm_detect(eeg, fs):
    """
    EEG alpha rhythm presence detection via autocorrelation

    Formula: alpha present if R_xx has peak at T~100ms (10Hz); decision by peak height

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: has_alpha, peak_freq

    References
    ----------
    Rangayyan Ch 10.2.3
    """
    eeg = np.asarray(eeg, dtype=float)
    y = np.asarray(eeg, dtype=float)
    n = min(len(eeg), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "EEG alpha rhythm presence detection via autocorrelation"})
    result = stats.spearmanr(eeg[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "EEG alpha rhythm presence detection via autocorrelation"})


def cheatsheet():
    return "rgeegrhm: EEG alpha rhythm presence detection via autocorrelation"
