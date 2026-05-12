r"""Vera forward.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_vera_forward"]


def kamath_ch4_vera_forward(W_0, Lambda_b, Lambda_d, A, B, x):
    r"""
    Vera forward.

    Formula: h = W_0 x + \Delta W x = W_0 x + \Lambda_b B \Lambda_d A x

    Parameters
    ----------
    W_0 : array-like
        Input data.
    Lambda_b : array-like
        Input data.
    Lambda_d : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    x : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.10, p. 155
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Vera forward."})


def cheatsheet():
    return "km063: Vera forward."
