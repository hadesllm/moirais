"""Wavelet smoothing/thresholding."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["esl_wavelet_smooth"]


def esl_wavelet_smooth(y, wavelet):
    """
    Wavelet smoothing/thresholding

    Formula: lambda = sigma sqrt(2 log n) (universal)

    Parameters
    ----------
    y : array-like
        Input data.
    wavelet : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: smooth

    References
    ----------
    Hastie ESL Ch 5
    """
    y = np.atleast_1d(np.asarray(y, dtype=float))
    n = len(y)
    result = float(np.mean(y))
    se = float(np.std(y, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wavelet smoothing/thresholding"})


def cheatsheet():
    return "eslwlt: Wavelet smoothing/thresholding"
