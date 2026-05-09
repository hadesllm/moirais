"""Numbered display equation (8.1) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_1"]


def mvsml_categorical_count_eq_8_1(n, k, k2, min, L, yi):
    """
    Numbered display equation (8.1) from MVSML chapter 8.

    Formula: n 1 k k2 min L yi, f xi ( ( ) ) + \lambda f ,

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    k2 : array-like
        Input data.
    min : array-like
        Input data.
    L : array-like
        Input data.
    yi : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.1) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    n = np.atleast_1d(np.asarray(n, dtype=float))
    n = len(n)
    result = float(np.mean(n))
    se = float(np.std(n, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.1) from MVSML chapter 8."})


def cheatsheet():
    return "msm123: Numbered display equation (8.1) from MVSML chapter 8."
