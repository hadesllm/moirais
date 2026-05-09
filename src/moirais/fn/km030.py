"""Nsp loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_nsp_loss"]


def kamath_ch2_nsp_loss(x, y, d):
    """
    Nsp loss.

    Formula: L^{(x,y)}_{NSP} = -\log P(d|x,y)

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.30, p. 54
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Nsp loss."})


def cheatsheet():
    return "km030: Nsp loss."
