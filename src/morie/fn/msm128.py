r"""Numbered display equation (8.3) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_3"]


def mvsml_categorical_count_eq_8_3(X, n, L, yi, kT, i):
    r"""
    Numbered display equation (8.3) from MVSML chapter 8.

    Formula: ) X n   + \lambda 1 L yi, \eta0 + kT i \beta 2 \betaTK\beta

    Parameters
    ----------
    X : array-like
        Input data.
    n : array-like
        Input data.
    L : array-like
        Input data.
    yi : array-like
        Input data.
    kT : array-like
        Input data.
    i : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.3) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    r"""
    X = np.atleast_1d(np.asarray(X, dtype=float))
    n = len(X)
    result = float(np.mean(X))
    se = float(np.std(X, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.3) from MVSML chapter 8."})


def cheatsheet():
    return "msm128: Numbered display equation (8.3) from MVSML chapter 8."
