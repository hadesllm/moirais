r"""Smd.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_smd"]


def kamath_ch8_smd(x, y, E):
    r"""
    Smd.

    Formula: \mathrm{SMD}(x^n,y^n) = \|E(x_1^{l_x}) - E(y_1^{l_y})\|

    Parameters
    ----------
    x : array-like
        Input data.
    y : array-like
        Input data.
    E : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.14, p. 326
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Smd."})


def cheatsheet():
    return "km126: Smd."
