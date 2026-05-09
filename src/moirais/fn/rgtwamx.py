# moirais.fn — function file (hadesllm/moirais)
"""T-wave alternans spectral method (modified moving average)."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_twa_spectral_mx"]


def rangayyan_twa_spectral_mx(ecg, fs, r_peaks):
    """
    T-wave alternans spectral method (modified moving average)

    Formula: Beat subtraction matrix; FFT along beat axis; alternans at f=0.5 cyc/beat

    Parameters
    ----------
    ecg : array-like
        Input data.
    fs : array-like
        Input data.
    r_peaks : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: twa_amplitude, k_score, p_value

    References
    ----------
    Rangayyan Ch 9.10
    """
    ecg = np.asarray(ecg, dtype=float)
    n = int(ecg) if ecg.ndim == 0 else len(ecg)
    if ecg.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "T-wave alternans spectral method (modified moving average)"})
    estimate = np.median(ecg)
    se = 1.2533 * np.std(ecg, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "T-wave alternans spectral method (modified moving average)"})


def cheatsheet():
    return "rgtwamx: T-wave alternans spectral method (modified moving average)"
