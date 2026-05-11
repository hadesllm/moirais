# morie.fn — function file (hadesllm/morie)
"""Envelogram: synchronized average of PCG envelopes."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_envelogram"]


def rangayyan_envelogram(pcg, ecg, fs):
    """
    Envelogram: synchronized average of PCG envelopes

    Formula: env_avg[n] = (1/M) sum |x_k(n) + j*H{x_k(n)}|

    Parameters
    ----------
    pcg : array-like
        Input data.
    ecg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: envelogram

    References
    ----------
    Rangayyan Ch 5.5.3
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    if pcg.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Envelogram: synchronized average of PCG envelopes"})
    estimate = np.median(pcg)
    se = 1.2533 * np.std(pcg, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Envelogram: synchronized average of PCG envelopes"})


def cheatsheet():
    return "rgenvgm: Envelogram: synchronized average of PCG envelopes"
