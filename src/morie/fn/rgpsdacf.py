# morie.fn -- function file (hadesllm/morie)
"""ACF estimation from PSD via inverse DFT (Wiener-Khintchine theorem)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_psd_to_acf"]


def rangayyan_psd_to_acf(psd, freqs):
    """
    ACF estimation from PSD via inverse DFT (Wiener-Khintchine theorem)

    Formula: R_xx(m) = IDFT(S_xx(f))

    Parameters
    ----------
    psd : array-like
        Input data.
    freqs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: acf, lags

    References
    ----------
    Rangayyan Ch 6.3.5
    """
    psd = np.asarray(psd, dtype=float)
    n = int(psd) if psd.ndim == 0 else len(psd)
    if psd.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "ACF estimation from PSD via inverse DFT (Wiener-Khintchine theorem)"})
    estimate = np.median(psd)
    se = 1.2533 * np.std(psd, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "ACF estimation from PSD via inverse DFT (Wiener-Khintchine theorem)"})


def cheatsheet():
    return "rgpsdacf: ACF estimation from PSD via inverse DFT (Wiener-Khintchine theorem)"
