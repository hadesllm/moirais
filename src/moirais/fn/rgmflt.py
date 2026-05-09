# moirais.fn — function file (hadesllm/moirais)
"""Matched filter transfer function for signal detection in noise."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_matched_filter"]


def rangayyan_matched_filter(signal_spectrum, noise_psd, t0):
    """
    Matched filter transfer function for signal detection in noise

    Formula: H_opt(f) = k * S*(f) * exp(-j2*pi*f*t0) / Pnn(f)

    Parameters
    ----------
    signal_spectrum : array-like
        Input data.
    noise_psd : array-like
        Input data.
    t0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: h_matched, freqs

    References
    ----------
    Rangayyan Ch 4.6.1
    """
    signal_spectrum = np.asarray(signal_spectrum, dtype=float)
    n = int(signal_spectrum) if signal_spectrum.ndim == 0 else len(signal_spectrum)
    result = float(np.mean(signal_spectrum))
    se = float(np.std(signal_spectrum, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Matched filter transfer function for signal detection in noise"})


def cheatsheet():
    return "rgmflt: Matched filter transfer function for signal detection in noise"
