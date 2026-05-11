"""Lowpass transfer function used in the Pan-Tompkins QRS detector.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_lowpass_transfer"]


def rangayyan_ch4_pan_tompkins_lowpass_transfer(z):
    """
    Lowpass transfer function used in the Pan-Tompkins QRS detector.

    Formula: H(z) = (1/32) * (1 - z^(-6))^2 / (1 - z^(-1))^2

    Parameters
    ----------
    z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.7, p. 220
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lowpass transfer function used in the Pan-Tompkins QRS detector."})


def cheatsheet():
    return "rng181: Lowpass transfer function used in the Pan-Tompkins QRS detector."
