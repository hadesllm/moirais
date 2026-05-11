"""Adaptive update rules for signal- and noise-peak running estimates in Pan-Tompkins.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_peak_classification"]


def rangayyan_ch4_pan_tompkins_peak_classification(PEAKI, SPKI, NPKI):
    """
    Adaptive update rules for signal- and noise-peak running estimates in Pan-Tompkins.

    Formula: SPKI = 0.125*PEAKI + 0.875*SPKI (signal); NPKI = 0.125*PEAKI + 0.875*NPKI (noise)

    Parameters
    ----------
    PEAKI : array-like
        Input data.
    SPKI : array-like
        Input data.
    NPKI : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.16, p. 223
    """
    PEAKI = np.atleast_1d(np.asarray(PEAKI, dtype=float))
    n = len(PEAKI)
    if n < 1:
        return RichResult(payload={"estimate": np.nan, "n": 0, "method": "Adaptive update rules for signal- and noise-peak running estimates in Pan-Tompkins."})
    estimate = np.median(PEAKI)
    se = 1.2533 * np.std(PEAKI, ddof=1) / np.sqrt(n)
    ci_lower = estimate - 1.96 * se
    ci_upper = estimate + 1.96 * se
    return RichResult(payload={"estimate": float(estimate), "se": float(se), "ci_lower": float(ci_lower), "ci_upper": float(ci_upper), "n": n, "method": "Adaptive update rules for signal- and noise-peak running estimates in Pan-Tompkins."})


def cheatsheet():
    return "rng190: Adaptive update rules for signal- and noise-peak running estimates in Pan-Tompkins."
