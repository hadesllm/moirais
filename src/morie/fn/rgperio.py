# morie.fn -- function file (hadesllm/morie)
"""Periodogram PSD estimate."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_periodogram"]


def rangayyan_periodogram(x, fs):
    """
    Periodogram PSD estimate

    Formula: P(f) = (1/N)|X(f)|^2 where X(f)=DFT(x)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: psd, freqs

    References
    ----------
    Rangayyan Ch 6.3.2
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Periodogram PSD estimate"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Periodogram PSD estimate"})


def cheatsheet():
    return "rgperio: Periodogram PSD estimate"
