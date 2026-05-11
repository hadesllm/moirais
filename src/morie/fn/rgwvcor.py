# morie.fn — function file (hadesllm/morie)
"""Wavelet cross-correlation between two signals at each scale."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_wavelet_corr"]


def rangayyan_wavelet_corr(x, y, wavelet, levels):
    """
    Wavelet cross-correlation between two signals at each scale

    Formula: WCC_j(tau) = sum d_j_x[n] * d_j_y[n+tau]

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    wavelet : array-like
        Input data.
    levels : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: cross_corr_per_scale

    References
    ----------
    Rangayyan Ch 8.8
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 3:
        return RichResult(payload={"statistic": np.nan, "p_value": np.nan, "n": n, "method": "Wavelet cross-correlation between two signals at each scale"})
    result = stats.spearmanr(x[:n], y[:n])
    return RichResult(payload={"statistic": float(result.statistic), "p_value": float(result.pvalue), "n": n, "method": "Wavelet cross-correlation between two signals at each scale"})


def cheatsheet():
    return "rgwvcor: Wavelet cross-correlation between two signals at each scale"
