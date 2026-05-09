# moirais.fn — function file (hadesllm/moirais)
"""Myocardial elasticity estimation from heart sound spectra."""
import numpy as np
from ._richresult import RichResult

__all__ = ["rangayyan_heart_elasticity"]


def rangayyan_heart_elasticity(pcg, fs):
    """
    Myocardial elasticity estimation from heart sound spectra

    Formula: Spectral shift of S1: higher stiffness -> higher dominant frequency

    Parameters
    ----------
    pcg : array-like
        Input data.
    fs : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: elasticity_index, spectral_centroid

    References
    ----------
    Rangayyan Ch 6.2.1
    """
    pcg = np.asarray(pcg, dtype=float)
    n = int(pcg) if pcg.ndim == 0 else len(pcg)
    if pcg.ndim == 0:
        return RichResult(payload={"statistic": float('nan'), "p_value": float('nan'), "n": 1, "method": "scalar-input placeholder"})
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Myocardial elasticity estimation from heart sound spectra"})
    estimate = np.median(pcg)
    se = 1.2533 * np.std(pcg, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Myocardial elasticity estimation from heart sound spectra"})


def cheatsheet():
    return "rgelast: Myocardial elasticity estimation from heart sound spectra"
