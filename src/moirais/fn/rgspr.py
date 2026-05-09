# moirais.fn — function file (hadesllm/moirais)
"""Spectral power ratio (LF/HF) for HRV analysis."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_spectral_power_ratio"]


def rangayyan_spectral_power_ratio(rr_psd, freqs):
    """
    Spectral power ratio (LF/HF) for HRV analysis

    Formula: LF = integral_{0.04}^{0.15} S(f)df; HF = integral_{0.15}^{0.40} S(f)df; ratio=LF/HF

    Parameters
    ----------
    rr_psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: lf, hf, lf_hf_ratio

    References
    ----------
    Rangayyan Ch 6.4.2
    """
    rr_psd = np.asarray(rr_psd, dtype=float)
    n = int(rr_psd) if rr_psd.ndim == 0 else len(rr_psd)
    result = float(np.mean(rr_psd))
    se = float(np.std(rr_psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Spectral power ratio (LF/HF) for HRV analysis"})


def cheatsheet():
    return "rgspr: Spectral power ratio (LF/HF) for HRV analysis"
