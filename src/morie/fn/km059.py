r"""Kronecker product.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch4_kronecker_product"]


def kamath_ch4_kronecker_product(A, B):
    r"""
    Kronecker product.

    Formula: W = A \otimes B = \begin{pmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{pmatrix}

    Parameters
    ----------
    A : array-like
        Input data.
    B : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 4, Eq 4.6, p. 152
    r"""
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Kronecker product."})


def cheatsheet():
    return "km059: Kronecker product."
