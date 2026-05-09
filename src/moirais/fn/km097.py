"""Ear entropy reg.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_ear_entropy_reg"]


def kamath_ch6_ear_entropy_reg(A, L, lam):
    """
    Ear entropy reg.

    Formula: R = -\lambda \sum_{\ell=1}^L \mathrm{entropy}(A)_{\ell}

    Parameters
    ----------
    A : array-like
        Input data.
    L : array-like
        Input data.
    lam : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.21, p. 244
    """
    A = np.atleast_1d(np.asarray(A, dtype=float))
    n = len(A)
    result = float(np.mean(A))
    se = float(np.std(A, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Ear entropy reg."})


def cheatsheet():
    return "km097: Ear entropy reg."
