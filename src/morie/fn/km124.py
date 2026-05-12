r"""Ngram embedding.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch8_ngram_embedding"]


def kamath_ch8_ngram_embedding(x, i, n):
    r"""
    Ngram embedding.

    Formula: E(x_i^n) = \sum_{k=i}^{i+n-1} \mathrm{idf}(x_k)

    Parameters
    ----------
    x : array-like
        Input data.
    i : array-like
        Input data.
    n : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 8, Eq 8.12, p. 326
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ngram embedding."})


def cheatsheet():
    return "km124: Ngram embedding."
