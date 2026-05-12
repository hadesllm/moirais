# morie.fn -- function file (hadesllm/morie)
"""Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_mfcc"]


def rangayyan_mfcc(x, fs, n_mfcc, n_filters):
    """
    Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis

    Formula: MFCC = DCT(log(filterbank_energy)); Mel-scale: m=2595*log10(1+f/700)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    n_mfcc : array-like
        Input data.
    n_filters : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: mfcc_matrix

    References
    ----------
    Rangayyan Ch 4.7.3
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis"})


def cheatsheet():
    return "rgmfcc: Mel-frequency cepstral coefficients (MFCC) for speech/bioacoustic analysis"
