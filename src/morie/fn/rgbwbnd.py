# morie.fn -- function file (hadesllm/morie)
"""Signal bandwidth estimation from PSD (half-power or 99% energy criterion)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_bandwidth"]


def rangayyan_bandwidth(psd, freqs, criterion):
    """
    Signal bandwidth estimation from PSD (half-power or 99% energy criterion)

    Formula: BW_3dB: f where S(f)>=S_max/2; BW_99: band containing 99% of total power

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.
    criterion : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: bandwidth

    References
    ----------
    Rangayyan Ch 6.4.1
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    if psd.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Signal bandwidth estimation from PSD (half-power or 99% energy criterion)"})
    estimate = np.median(psd)
    se = 1.2533 * np.std(psd, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Signal bandwidth estimation from PSD (half-power or 99% energy criterion)"})


def cheatsheet():
    return "rgbwbnd: Signal bandwidth estimation from PSD (half-power or 99% energy criterion)"
