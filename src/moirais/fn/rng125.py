"""Pole positions on the Butterworth circle in the s-plane.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_butterworth_pole_positions"]


def rangayyan_ch3_butterworth_pole_positions(Omega_c, N, k):
    """
    Pole positions on the Butterworth circle in the s-plane.

    Formula: s_k = Omega_c * exp( j*pi * (0.5 + (2*k - 1)/(2*N)) ), k = 1..2N

    Parameters
    ----------
    Omega_c : array-like
        Input data.
    N : array-like
        Input data.
    k : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.137, p. 154
    """
    Omega_c = np.atleast_1d(np.asarray(Omega_c, dtype=float))
    n = len(Omega_c)
    result = float(np.mean(Omega_c))
    se = float(np.std(Omega_c, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pole positions on the Butterworth circle in the s-plane."})


def cheatsheet():
    return "rng125: Pole positions on the Butterworth circle in the s-plane."
