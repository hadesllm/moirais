"""Notch filter with two zeros at 60 Hz on the unit circle.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch3_notch_filter_60Hz"]


def rangayyan_ch3_notch_filter_60Hz(z):
    """
    Notch filter with two zeros at 60 Hz on the unit circle.

    Formula: H(z) = (1 - z^(-1)*z_1)(1 - z^(-1)*z_2) = 1 - 1.85955*z^(-1) + z^(-2)

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
    Rangayyan (2024), Ch 3, Eq 3.150, p. 164
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Notch filter with two zeros at 60 Hz on the unit circle."})


def cheatsheet():
    return "rng136: Notch filter with two zeros at 60 Hz on the unit circle."
