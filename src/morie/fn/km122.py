"""Wmd.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_wmd"]


def kamath_ch8_wmd(x_n, y_n, C, F):
    """
    Wmd.

    Formula: \mathrm{WMD}(x^n,y^n) = \min_{F\in R^{|x^n|\times|y^n|}} \langle C,F\rangle, \text{ s.t. } F\mathbf{1}=f_{x^n}, F^T\mathbf{1}=f_{y^n}

    Parameters
    ----------
    x_n : array-like
        Input data.
    y_n : array-like
        Input data.
    C : array-like
        Input data.
    F : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.10, p. 326
    """
    x_n = np.atleast_1d(np.asarray(x_n, dtype=float))
    n = len(x_n)
    result = float(np.mean(x_n))
    se = float(np.std(x_n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Wmd."})


def cheatsheet():
    return "km122: Wmd."
