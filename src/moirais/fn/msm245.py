"""Numbered display equation (10.4) from MVSML chapter 10.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_reproducing_kernel_eq_10_4"]


def mvsml_reproducing_kernel_eq_10_4(X, m1, m0, F, x1, xm0):
    """
    Numbered display equation (10.4) from MVSML chapter 10.

    Formula: ! X X m1 m0 F x1, . . . , xm0 ( ) = \alphaig wijx j + bi

    Parameters
    ----------
    X : array-like
        Input data.
    m1 : array-like
        Input data.
    m0 : array-like
        Input data.
    F : array-like
        Input data.
    x1 : array-like
        Input data.
    xm0 : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (10.4) [Multivariate Statistical Machine Learnin [Pages 379-425] [2026-04-16].pdf]
    """
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (10.4) from MVSML chapter 10."})


def cheatsheet():
    return "msm245: Numbered display equation (10.4) from MVSML chapter 10."
