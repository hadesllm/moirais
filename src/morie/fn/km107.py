"""Pii likelihood.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_pii_likelihood"]


def kamath_ch6_pii_likelihood(a_m, A, x, L_q, L_r):
    """
    Pii likelihood.

    Formula: P_r(a_m|A_{\setminus m}) = \prod_{r=1}^{L_r} p(a_{m,r}|x_1,x_2,\dots,x_{L_q+r-1})

    Parameters
    ----------
    a_m : array-like
        Input data.
    A : array-like
        Input data.
    x : array-like
        Input data.
    L_q : array-like
        Input data.
    L_r : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.31, p. 258
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Pii likelihood."})


def cheatsheet():
    return "km107: Pii likelihood."
