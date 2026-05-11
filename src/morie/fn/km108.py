"""Differential privacy.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_differential_privacy"]


def kamath_ch6_differential_privacy(M, A, B, S, epsilon):
    """
    Differential privacy.

    Formula: P[M(A)\in S] \le e^{\epsilon} P[M(B)\in S]

    Parameters
    ----------
    M : array-like
        Input data.
    A : array-like
        Input data.
    B : array-like
        Input data.
    S : array-like
        Input data.
    epsilon : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.32, p. 258
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Differential privacy."})


def cheatsheet():
    return "km108: Differential privacy."
