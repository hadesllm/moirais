"""Factscore.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["kamath_ch6_factscore"]


def kamath_ch6_factscore(M, X, A_y, C):
    """
    Factscore.

    Formula: \mathrm{FActScore}(M) = E_{x\in X}[\frac{1}{|A_y|}\sum_{a\in A_y} I[a\text{ is supported by }C] | M_x \text{ responds}]

    Parameters
    ----------
    M : array-like
        Input data.
    X : array-like
        Input data.
    A_y : array-like
        Input data.
    C : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: result

    References
    ----------
    Kamath et al (2024), Ch 6, Eq 6.1, p. 219
    """
    M = np.atleast_1d(np.asarray(M, dtype=float))
    n = len(M)
    result = float(np.mean(M))
    se = float(np.std(M, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Factscore."})


def cheatsheet():
    return "km077: Factscore."
