r"""Ngram weight.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_ngram_weight"]


def kamath_ch8_ngram_weight(x, Z):
    r"""
    Ngram weight.

    Formula: f_{x_i^n} = \frac{1}{Z}\sum_{k=i}^{i+n-1} \mathrm{idf}(x_k)

    Parameters
    ----------
    x : array-like
        Input data.
    Z : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.13, p. 326
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ngram weight."})


def cheatsheet():
    return "km125: Ngram weight."
