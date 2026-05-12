r"""Multihead concat.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch2_multihead_concat"]


def kamath_ch2_multihead_concat(heads, W_O):
    r"""
    Multihead concat.

    Formula: \mathrm{multihead}(Q,K,V) = W^O\,\mathrm{concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)

    Parameters
    ----------
    heads : array-like
        Input data.
    W_O : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 2, Eq 2.16, p. 36
    r"""
    heads = np.atleast_1d(np.asarray(heads, dtype=float))
    n = len(heads)
    result = float(np.mean(heads))
    se = float(np.std(heads, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Multihead concat."})


def cheatsheet():
    return "km016: Multihead concat."
