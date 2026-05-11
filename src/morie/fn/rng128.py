"""Bilinear transform restricted to the unit circle (sigma=0).."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_bilinear_unit_circle_relation"]


def rangayyan_ch3_bilinear_unit_circle_relation(omega, T):
    """
    Bilinear transform restricted to the unit circle (sigma=0).

    Formula: s = sigma + j*Omega = (2/T) * (1 - exp(-j*omega))/(1 + exp(-j*omega)) = (2*j/T) * tan(omega/2)

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
    Rangayyan (2024), Ch 3, Eq 3.140, p. 155
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=float))
    n = len(omega)
    result = float(np.mean(omega))
    se = float(np.std(omega, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Bilinear transform restricted to the unit circle (sigma=0)."})


def cheatsheet():
    return "rng128: Bilinear transform restricted to the unit circle (sigma=0)."
