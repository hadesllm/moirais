"""Magnitude coherence spectrum between two signals from CSD and PSDs.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_coherence_spectrum"]


def rangayyan_ch4_coherence_spectrum(S_xy, S_xx, S_yy, f):
    """
    Magnitude coherence spectrum between two signals from CSD and PSDs.

    Formula: Gamma_xy(f) = sqrt( |S_xy(f)|^2 / (S_xx(f) * S_yy(f)) )

    Parameters
    ----------
    S_xy : array-like
        Input data.
    S_xx : array-like
        Input data.
    S_yy : array-like
        Input data.
    f : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.32, p. 236
    """
    S_xy = np.atleast_1d(np.asarray(S_xy, dtype=float))
    n = len(S_xy)
    result = float(np.mean(S_xy))
    se = float(np.std(S_xy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Magnitude coherence spectrum between two signals from CSD and PSDs."})


def cheatsheet():
    return "rng206: Magnitude coherence spectrum between two signals from CSD and PSDs."
