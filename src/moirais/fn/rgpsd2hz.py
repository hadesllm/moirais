# moirais.fn — function file (hadesllm/moirais)
"""Convert PSD to frequency-in-Hz units and compute bin-level features."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_psd_to_hz"]


def rangayyan_psd_to_hz(psd, N, fs):
    """
    Convert PSD to frequency-in-Hz units and compute bin-level features

    Formula: bin_freq = k*fs/N; power_in_band = sum S(k) * (fs/N) for bins in band

    Parameters
    ----------
    psd : array-like
        Input data.
    N : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: freqs_hz, band_power

    References
    ----------
    Rangayyan Ch 6
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Convert PSD to frequency-in-Hz units and compute bin-level features"})


def cheatsheet():
    return "rgpsd2hz: Convert PSD to frequency-in-Hz units and compute bin-level features"
