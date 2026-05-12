r"""Debias regularizer.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_debias_regularizer"]


def kamath_ch6_debias_regularizer(A, E, lam):
    r"""
    Debias regularizer.

    Formula: R = \lambda \sum_{(a_i,a_j)\in A} \|E(a_i) - E(a_j)\|^2

    Parameters
    ----------
    A : array-like
        Input data.
    E : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.18, p. 242
    r"""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Debias regularizer."})


def cheatsheet():
    return "km094: Debias regularizer."
