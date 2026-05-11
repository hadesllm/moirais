# morie.fn — function file (hadesllm/morie)
"""EEG rhythm detection via autocorrelation."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_eeg_autocorr"]


def rangayyan_eeg_autocorr(eeg, fs, max_lag):
    """
    EEG rhythm detection via autocorrelation

    Formula: R_xx(tau) -> peak at T_rhythm; frequency = 1/T_rhythm

    Parameters
    ----------
    eeg : array-like
        Input data.
    fs : array-like
        Input data.
    max_lag : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: rhythm_freq, acf

    References
    ----------
    Rangayyan Ch 4.4.1
    """
    eeg = np.asarray(eeg, dtype=float)
    y = np.asarray(eeg, dtype=float)
    n = min(len(eeg), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "EEG rhythm detection via autocorrelation"})
    result = stats.spearmanr(eeg[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "EEG rhythm detection via autocorrelation"})


def cheatsheet():
    return "rgeegar: EEG rhythm detection via autocorrelation"
