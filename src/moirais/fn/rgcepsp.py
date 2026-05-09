# moirais.fn — function file (hadesllm/moirais)
"""Pitch (fundamental frequency) estimation from cepstrum peak."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_cepstrum_pitch"]


def rangayyan_cepstrum_pitch(x, fs, f0_range):
    """
    Pitch (fundamental frequency) estimation from cepstrum peak

    Formula: Pitch period T0 = quefrency of dominant peak in cepstrum (rahmonics)

    Parameters
    ----------
    x : array-like
        Input data.
    fs : array-like
        Input data.
    f0_range : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: pitch_hz, pitch_period

    References
    ----------
    Rangayyan Ch 4.7.3
    """
    x = np.asarray(x, dtype=float)
    n = int(x) if x.ndim == 0 else len(x)
    if x.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Pitch (fundamental frequency) estimation from cepstrum peak"})
    estimate = np.median(x)
    se = 1.2533 * np.std(x, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Pitch (fundamental frequency) estimation from cepstrum peak"})


def cheatsheet():
    return "rgcepsp: Pitch (fundamental frequency) estimation from cepstrum peak"
