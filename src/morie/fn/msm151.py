"""Numbered display equation (8.11) from MVSML chapter 8.."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["mvsml_categorical_count_eq_8_11"]


def mvsml_categorical_count_eq_8_11(m, US, used, where, U, are):
    """
    Numbered display equation (8.11) from MVSML chapter 8.

    Formula: m,m = US 2 1=2S 2 1=2U0 is used where U are Therefore, an eigen decomposition of K 2 1 the eigenvectors of order m  m and Sm,m is a diagonal matrix of order m  m with the eigenvalues ordered from largest to smallest. These values are substituted in Q  resulting in un  N 0, \sigma2 uKn,mUS 2 1=2S 2 1=2U0K0 ), and thus, thanks to the proper- n,m ties of the normal distribution, model (8.8) can be expressed like model

    Parameters
    ----------
    m : array-like
        Input data.
    US : array-like
        Input data.
    used : array-like
        Input data.
    where : array-like
        Input data.
    U : array-like
        Input data.
    are : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: expression

    References
    ----------
    MVSML, Eq. (8.11) [Multivariate Statistical Machine Learnin [Pages 251-336] [2026-04-16].pdf]
    """
    m = np.atleast_1d(np.asarray(m, dtype=float))
    n = len(m)
    result = float(np.mean(m))
    se = float(np.std(m, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Numbered display equation (8.11) from MVSML chapter 8."})


def cheatsheet():
    return "msm151: Numbered display equation (8.11) from MVSML chapter 8."
