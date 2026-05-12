r"""Rtd loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_rtd_loss"]


def kamath_ch2_rtd_loss(xhat, d):
    r"""
    Rtd loss.

    Formula: L^{(x)}_{RTD} = -\frac{1}{|\hat{x}|}\sum_{i=1}^{|\hat{x}|} \log P(d|\hat{x}_i)

    Parameters
    ----------
    xhat : array-like
        Input data.
    d : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.23, p. 51
    r"""
    xhat = np.atleast_1d(np.asarray(xhat, dtype=float))
    n = len(xhat)
    result = float(np.mean(xhat))
    se = float(np.std(xhat, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Rtd loss."})


def cheatsheet():
    return "km023: Rtd loss."
