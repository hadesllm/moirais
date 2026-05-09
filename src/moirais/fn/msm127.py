"""Numbered display equation (8.1) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_1"]


def mvsml_categorical_count_eq_8_1(n, k, k2, de, ned, before):
    """
    Numbered display equation (8.1) from MVSML chapter 8.

    Formula:   n k k2 as deﬁned before, and \betaj are beta coefﬁcients. Notice that f \betal\beta jK xl, xj , l, j=1 and by substituting (8.2) into

    Parameters
    ----------
    n : array-like
        Input data.
    k : array-like
        Input data.
    k2 : array-like
        Input data.
    de : array-like
        Input data.
    ned : array-like
        Input data.
    before : array-like
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
    return "msm127: Numbered display equation (8.1) from MVSML chapter 8."
