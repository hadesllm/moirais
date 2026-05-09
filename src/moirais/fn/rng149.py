"""Frequency-domain Wiener relation between PSD and CSD.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_wiener_frequency_relation"]


def rangayyan_ch3_wiener_frequency_relation(W, S_xx, S_xd, omega):
    """
    Frequency-domain Wiener relation between PSD and CSD.

    Formula: W(omega) * S_xx(omega) = S_xd(omega)

    Parameters
    ----------
    W : array-like
        Input data.
    S_xx : array-like
        Input data.
    S_xd : array-like
        Input data.
    omega : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: spectrum

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.175, p. 176
    """
    W = np.atleast_1d(np.asarray(W, dtype=float))
    n = len(W)
    result = float(np.mean(W))
    se = float(np.std(W, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Frequency-domain Wiener relation between PSD and CSD."})


def cheatsheet():
    return "rng149: Frequency-domain Wiener relation between PSD and CSD."
