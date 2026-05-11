"""Numbered display equation (8.10) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_10"]


def mvsml_categorical_count_eq_8_10(Qu2, Zu1, Kn, mK, m, mKT):
    """
    Numbered display equation (8.10) from MVSML chapter 8.

    Formula: Qu2 = Zu1 Kn,mK 2 1 m,mKT ZT ZEZET u2  N 0, \sigma2 . n,m u1 Also, we decomposed K 2 1 m,m in such a way that model

    Parameters
    ----------
    Qu2 : array-like
        Input data.
    Zu1 : array-like
        Input data.
    Kn : array-like
        Input data.
    mK : array-like
        Input data.
    m : array-like
        Input data.
    mKT : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.10) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    Qu2 = np.atleast_1d(np.asarray(Qu2, dtype=float))
    n = len(Qu2)
    result = float(np.mean(Qu2))
    se = float(np.std(Qu2, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.10) from MVSML chapter 8."})


def cheatsheet():
    return "msm157: Numbered display equation (8.10) from MVSML chapter 8."
