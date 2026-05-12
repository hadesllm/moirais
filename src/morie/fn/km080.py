r"""Weat function.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_weat_function"]


def kamath_ch6_weat_function(A_1, A_2, W_1, W_2):
    r"""
    Weat function.

    Formula: f(A_1,A_2,W_1,W_2) = \sum_{a_1\in A_1} s(a_1,W_1,W_2) - \sum_{a_2\in A_2} s(a_2,W_1,W_2)

    Parameters
    ----------
    A_1 : array-like
        Input data.
    A_2 : array-like
        Input data.
    W_1 : array-like
        Input data.
    W_2 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.4, p. 234
    r"""
    A_1 = np.atleast_1d(np.asarray(A_1, dtype=float))
    n = len(A_1)
    result = float(np.mean(A_1))
    se = float(np.std(A_1, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Weat function."})


def cheatsheet():
    return "km080: Weat function."
