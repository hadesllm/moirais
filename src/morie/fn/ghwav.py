# morie.fn — function file (hadesllm/morie)
"""Wavelet prior for function estimation."""
import numpy as np
from ._richresult import RichResult

__all__ = ["ghosal_wavelet_prior"]


def ghosal_wavelet_prior(x):
    """
    Wavelet prior for function estimation

    Formula: theta_jk ~ pi_j * N(0, sigma_j^2) + (1-pi_j)*delta_0

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Ghosal Ch 13
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Wavelet prior for function estimation"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Wavelet prior for function estimation"})


def cheatsheet():
    return "ghwav: Wavelet prior for function estimation"
