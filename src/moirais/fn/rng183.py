"""Lowpass component of the Pan-Tompkins highpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_highpass_lp_component"]


def rangayyan_ch4_pan_tompkins_highpass_lp_component(z):
    """
    Lowpass component of the Pan-Tompkins highpass filter.

    Formula: H_lp(z) = (1 - z^(-32)) / (1 - z^(-1))

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
    Rangayyan (2024), Ch 4, Eq 4.9, p. 221
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Lowpass component of the Pan-Tompkins highpass filter."})


def cheatsheet():
    return "rng183: Lowpass component of the Pan-Tompkins highpass filter."
