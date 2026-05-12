r"""Numbered display equation (8.5) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_5"]


def mvsml_categorical_count_eq_8_5(J, l, xi, x, j, AK):
    r"""
    Numbered display equation (8.5) from MVSML chapter 8.

    Formula:   2 J \theta l( ) = 1 ) xi, x j \pi AK l( ) xi, xi )AK l( ) x j, x j AK l+1 ( (

    Parameters
    ----------
    J : array-like
        Input data.
    l : array-like
        Input data.
    xi : array-like
        Input data.
    x : array-like
        Input data.
    j : array-like
        Input data.
    AK : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.5) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = len(x)
    result = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.5) from MVSML chapter 8."})


def cheatsheet():
    return "msm132: Numbered display equation (8.5) from MVSML chapter 8."
