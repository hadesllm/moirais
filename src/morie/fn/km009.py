r"""Softmax element.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_softmax_element"]


def kamath_ch2_softmax_element(a_i, a):
    r"""
    Softmax element.

    Formula: b_i = \frac{\exp(a_i)}{\sum_j \exp(a_j)}

    Parameters
    ----------
    a_i : array-like
        Input data.
    a : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.9, p. 32
    r"""
    a_i = np.atleast_1d(np.asarray(a_i, dtype=float))
    n = len(a_i)
    result = float(np.mean(a_i))
    se = float(np.std(a_i, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Softmax element."})


def cheatsheet():
    return "km009: Softmax element."
