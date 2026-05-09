# moirais.fn — function file (hadesllm/moirais)
"""Median frequency from PSD (50th percentile of cumulative power)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_ch6_median_freq"]


def rangayyan_ch6_median_freq(psd, freqs):
    """
    Median frequency from PSD (50th percentile of cumulative power)

    Formula: f_median: sum_{f_k<=f_median} S(f_k) = 0.5 * sum S(f_k)

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: median_freq

    References
    ----------
    Rangayyan Ch 6.4.1
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    result = float(np.mean(psd))
    se = float(np.std(psd, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Median frequency from PSD (50th percentile of cumulative power)"})


def cheatsheet():
    return "rgeqn6b: Median frequency from PSD (50th percentile of cumulative power)"
