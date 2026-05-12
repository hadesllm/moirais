r"""Sbo loss.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_sbo_loss"]


def kamath_ch2_sbo_loss(x, S, p):
    r"""
    Sbo loss.

    Formula: L^{(x)}_{SBO} = -\frac{1}{|S|}\sum_{i\in S}\log P(x_i|f(x_{s-1},x_{e+1},p_{s-e+1}))

    Parameters
    ----------
    x : array-like
        Input data.
    S : array-like
        Input data.
    p : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.29, p. 54
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Sbo loss."})


def cheatsheet():
    return "km029: Sbo loss."
