"""Bilinear frequency warping: discrete omega from analog Omega.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_bilinear_warping_Omega_to_omega"]


def rangayyan_ch3_bilinear_warping_Omega_to_omega(Omega, T):
    """
    Bilinear frequency warping: discrete omega from analog Omega.

    Formula: omega = 2 * atan(Omega*T/2)

    Parameters
    ----------
    Omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.142, p. 155
    """
    Omega = np.atleast_1d(np.asarray(Omega, dtype=float))
    n = len(Omega)
    result = float(np.mean(Omega))
    se = float(np.std(Omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilinear frequency warping: discrete omega from analog Omega."})


def cheatsheet():
    return "rng130: Bilinear frequency warping: discrete omega from analog Omega."
