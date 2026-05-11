"""Alm loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_alm_loss"]


def kamath_ch2_alm_loss(z, M):
    """
    Alm loss.

    Formula: L^{(z(x,y))}_{ALM} = -\frac{1}{|M|}\sum_{i\in M}\log P(z_i|z_{\setminus M})

    Parameters
    ----------
    z : array-like
        Input data.
    M : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.28, p. 53
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Alm loss."})


def cheatsheet():
    return "km028: Alm loss."
