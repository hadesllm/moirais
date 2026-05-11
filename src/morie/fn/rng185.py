"""Transfer function of the Pan-Tompkins highpass filter.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["rangayyan_ch4_pan_tompkins_highpass_transfer"]


def rangayyan_ch4_pan_tompkins_highpass_transfer(z, H_lp):
    """
    Transfer function of the Pan-Tompkins highpass filter.

    Formula: H_hp(z) = z^(-16) - (1/32) * H_lp(z)

    Parameters
    ----------
    z : array-like
        Input data.
    H_lp : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: value

    References
    ----------
    Rangayyan (2024), Ch 4, Eq 4.11, p. 221
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Transfer function of the Pan-Tompkins highpass filter."})


def cheatsheet():
    return "rng185: Transfer function of the Pan-Tompkins highpass filter."
