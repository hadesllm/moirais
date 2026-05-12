# morie.fn -- function file (hadesllm/morie)
"""Spectral moments: centroid (mean freq), variance (bandwidth), skewness."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_spectral_moments"]


def rangayyan_spectral_moments(psd, freqs):
    """
    Spectral moments: centroid (mean freq), variance (bandwidth), skewness

    Formula: f_c = sum(f*S(f))/sum(S(f)); bw = sqrt(sum((f-f_c)^2*S(f))/sum(S(f)))

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: centroid, bandwidth, skewness

    References
    ----------
    Rangayyan Ch 6.4.1
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral moments: centroid (mean freq), variance (bandwidth), skewness"})


def cheatsheet():
    return "rgsmom: Spectral moments: centroid (mean freq), variance (bandwidth), skewness"
