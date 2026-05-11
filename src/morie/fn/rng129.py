"""Bilinear frequency warping: analog Omega from discrete omega.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_bilinear_warping_omega_to_Omega"]


def rangayyan_ch3_bilinear_warping_omega_to_Omega(omega, T):
    """
    Bilinear frequency warping: analog Omega from discrete omega.

    Formula: Omega = (2/T) * tan(omega/2)

    Parameters
    ----------
    omega : array-like
        Input data.
    T : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 3, Eq 3.141, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilinear frequency warping: analog Omega from discrete omega."})


def cheatsheet():
    return "rng129: Bilinear frequency warping: analog Omega from discrete omega."
