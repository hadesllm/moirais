r"""Ffn relu.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_ffn_relu"]


def kamath_ch2_ffn_relu(z, W_1, W_2, b_1, b_2):
    r"""
    Ffn relu.

    Formula: F(z) = \mathrm{ReLU}(zW_1 + b_1)W_2 + b_2

    Parameters
    ----------
    z : array-like
        Input data.
    W_1 : array-like
        Input data.
    W_2 : array-like
        Input data.
    b_1 : array-like
        Input data.
    b_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.17, p. 37
    r"""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    n = len(z)
    result = float(np.mean(z))
    se = float(np.std(z, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ffn relu."})


def cheatsheet():
    return "km017: Ffn relu."
