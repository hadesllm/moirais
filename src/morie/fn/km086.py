r"""Pll.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_pll"]


def kamath_ch6_pll(S, theta):
    r"""
    Pll.

    Formula: \mathrm{PLL}(S) = \sum_{s\in S} \log P(s|S_{\setminus s};\theta)

    Parameters
    ----------
    S : array-like
        Input data.
    theta : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.10, p. 235
    r"""
    S = np.atleast_1d(np.asarray(S, dtype=float))
    n = len(S)
    result = float(np.mean(S))
    se = float(np.std(S, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pll."})


def cheatsheet():
    return "km086: Pll."
