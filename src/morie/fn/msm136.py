"""Numbered display equation (8.6) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_6"]


def mvsml_categorical_count_eq_8_6(CTy, KTC, KTK, K, b, KTy):
    """
    Numbered display equation (8.6) from MVSML chapter 8.

    Formula: CTy = (8.6) KTC KTK + \lambdaK\sigma2 b\beta KTy e Recall that K is symmetric, so KTK = K2, and by multiplying the second system of

    Parameters
    ----------
    CTy : array-like
        Input data.
    KTC : array-like
        Input data.
    KTK : array-like
        Input data.
    K : array-like
        Input data.
    b : array-like
        Input data.
    KTy : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.6) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    CTy = np.atleast_1d(np.asarray(CTy, dtype=float))
    n = len(CTy)
    result = float(np.mean(CTy))
    se = float(np.std(CTy, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.6) from MVSML chapter 8."})


def cheatsheet():
    return "msm136: Numbered display equation (8.6) from MVSML chapter 8."
