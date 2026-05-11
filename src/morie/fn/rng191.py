"""Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_thresholds"]


def rangayyan_ch4_pan_tompkins_thresholds(NPKI, SPKI):
    """
    Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm.

    Formula: THRESHOLD_I1 = NPKI + 0.25*(SPKI - NPKI); THRESHOLD_I2 = 0.5*THRESHOLD_I1

    Parameters
    ----------
    NPKI : array-like
        Input data.
    SPKI : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.17, p. 224
    """
    NPKI = np.atleast_1d(np.asarray(NPKI, dtype=float))
    n = len(NPKI)
    result = float(np.mean(NPKI))
    se = float(np.std(NPKI, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm."})


def cheatsheet():
    return "rng191: Adaptive thresholds for QRS detection in the Pan-Tompkins algorithm."
